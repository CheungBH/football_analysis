import copy
import os
import time
from collections import defaultdict
from utils.augmentations import letterbox
from utils.torch_utils import select_device
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from models.common import DetectMultiBackend
import argparse
import copy
from team_assigner import TeamAssigner
from analyser.analysis import AnalysisManager
import json
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_boxes

from visualize import plot_tracking
from tracker import BYTETracker
import config.config as config

team_colors = []

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/vol/datastore/zhangbh/Downloads/best.pt",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='/vol/datastore/zhangbh/Downloads/20241224_HKUCourt_SideFar1_5_2.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "--click_image",
        type=str,
        default='',
        help="Path to your input image.",
    )
    parser.add_argument(
        "--court_image",
        type=str,
        default='court_reference/soccer-field.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_video_path",
        type=str,
        default='demo_output.mp4',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.25,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.45,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--no_ball_tracker",
        action="store_true",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument(
        "--save_asset",
        action="store_true",
        help="Whether save model assets",
    )
    parser.add_argument(
        "--track_before_knn",
        action="store_true",
        help="Conduct tracking before KNN",
    )
    parser.add_argument(
        "--use_json",
        action="store_true",
        help="Load json for inference",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def preprocess(image, input_size, swap=(2, 0, 1), return_origin_size=False):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    if return_origin_size:
        start_y = 0
        padded_img = resized_img
    else:
    #padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        start_y = (input_size[0] - int(img.shape[0] * r)) // 2
        start_x = (input_size[1] - int(img.shape[1] * r)) // 2
        padded_img[start_y:start_y + int(img.shape[0] * r), start_x:start_x + int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r, start_y

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


class Predictor(object):
    def __init__(self, args):
        self.args = args
        # self.session = onnxruntime.InferenceSession(args.model)
        self.input_shape = tuple(map(int, args.input_shape.split(',')))
        self.device =  select_device(args.device)
        self.model = DetectMultiBackend(args.model, device=self.device, fp16=True)
        self.stride = self.model.stride

    def inference(self, ori_img):


        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        # img, ratio,start_y = preprocess(ori_img, self.input_shape, return_origin_size=True)
        # img = img.to(self.device)
        im = letterbox(ori_img, list(self.input_shape), stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im)
        pred = non_max_suppression(pred, self.args.score_thr, self.args.nms_thr, None, False, max_det=1000)
        # pred = pred[0]
        if pred is None:
            return None, img_info
        output = pred[0]
        output[:, :4] = scale_boxes(im.shape[2:], output[:, :4], ori_img.shape).round()
        output = clip_coords(output, ori_img.shape[:2])
        return output.detach().cpu(), img_info
        # img_info["ratio"] = ratio
        # ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        #
        # output = self.session.run(None, ort_inputs)
        # if len(output) > 1:
        #     output = [output[0]]
        # img = torch.from_numpy(img).to(self.device)
        # output = self.model(img[None, :, :, :])

        predictions = np.squeeze(output[0]).T

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2 - int(start_y)
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2 - int(start_y)
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        return dets, img_info

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    return boxes

def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.use_json:
        args.save_asset = False

    tv_h, tv_w = config.topview_height, config.topview_width

    vid_writer = cv2.VideoWriter(
        args.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    topview_writer = cv2.VideoWriter(
        "/".join(args.output_video_path.split("/")[:-1]) + "top_view.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (tv_h, tv_w)
    )
    # tracker = BYTETracker(args, frame_rate=30)
    frame_id = 0
    # team1_tracker = BYTETracker(args, frame_rate=30)
    # team2_tracker = BYTETracker(args, frame_rate=30)
    team_assigner = TeamAssigner()

    # results = []
    # pixel_points = [(485, 217), (820, 216), (894, 403), (416, 404)]
    # real_points = [(0, 0), (10, 0), (10, 23.77), (0, 23.77)]
    points = []

    def click_court():
        # global points

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(court_img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('click_court', court_img)
                # if len(points) > 4:
                #     points.pop(0)
                print(points)

        height, width, channel = frame.shape
        cv2.namedWindow("click_court", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("click_court", width, height)
        cv2.namedWindow("click_court")
        cv2.setMouseCallback("click_court", click_event)

        while True:
            cv2.imshow("click_court", court_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        print(points)

    def click_color():
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color = color_img[y, x]
                team_colors.append(color)
                print(color)
                # copied_frame = copy.deepcopy(frame)
                # cv2.circle(copied_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('click_color', color_img)

        height, width, channel = frame.shape
        cv2.namedWindow("click_color", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("click_color", width, height)
        cv2.namedWindow("click_color")
        cv2.setMouseCallback("click_color", click_event)

        while True:
            cv2.imshow("click_color", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                break

    # real_player_location = {defaultdict}
    top_view_img_tpl = cv2.imread(args.court_image)
    real_ball_history=[]
    team1_dict = defaultdict(list)
    team2_dict = defaultdict(list)
    goalkeeper_dict = defaultdict(list)
    referee_dict = defaultdict(list)

    analysis = AnalysisManager(config.check_action, ((0, 0)))

    while True:
        #referee_dict = defaultdict(list)
        top_view_img = copy.deepcopy(top_view_img_tpl)
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id == 0:
                if args.use_json:
                    json_path = ".".join(args.video_path.split(".")[:-1]) + '.json'
                    with open(json_path, 'r') as f:
                        assets = json.load(f)
                    court_points = np.array(assets["court_point"])
                    game_points = np.array(assets["game_point"])
                    matrix = np.array(assets["court_matrix"])
                    team_colors = {idx: np.array(color) for idx, color in enumerate(assets["team_colors"])}
                    # team_box_colors = team_colors
                    # outputs = np.array(assets[str(frame_id)])
                    team_assigner.assign_color()
                    # img_info = {"height": height, "width": width, "raw_img": frame}
                else:
                    color_img = cv2.imread(args.click_image) if args.click_image else frame
                    #click_color()
                    team_assigner.assign_color()

                    # team_colors = team_assigner.team_colors
                    # team_box_colors = team_assigner.team_colors
                    team_colors = {0: np.array([0, 0, 255], dtype=np.uint8),
                                   1: np.array([125, 125, 125], dtype=np.uint8),
                                   2: np.array([255, 0, 0], dtype=np.uint8), 3: np.array([0, 0, 0], dtype=np.uint8)}
                    team_box_colors = team_colors
                    trackers = [BYTETracker(args, frame_rate=30) for _ in range(len(team_box_colors))]
                    ball_tracker = BYTETracker(args, frame_rate=30)
                    print(team_colors)
                    court_img = copy.deepcopy(frame)
                    click_court()
                    game_points = np.array(points)
                    time.sleep(1)
                    court_img = cv2.imread(args.court_image)
                    points = []
                    click_court()
                    court_points = np.array(points)
                    matrix, _ = cv2.findHomography(game_points, court_points, cv2.RANSAC)

                    if args.save_asset:
                        asset_path = ".".join(args.video_path.split(".")[:-1]) + '.json'
                        assets = {
                            "court_matrix": matrix.tolist(),
                            "game_point": game_points.tolist(),
                            "court_point": court_points.tolist(),
                            "team_colors": [color.tolist() for index, color in team_colors.items()],
                        }
                    cv2.destroyAllWindows()

                if args.track_before_knn:
                    trackers = BYTETracker(args, frame_rate=30)
                else:
                    trackers = [BYTETracker(args, frame_rate=30) for _ in range(len(team_colors))]

                ball_tracker = BYTETracker(args, frame_rate=30)

            if args.use_json:
                outputs = np.array(assets[str(frame_id)])
                img_info = {"height": height, "width": width, "raw_img": frame}
            else:
                outputs, img_info = predictor.inference(frame)
                if args.save_asset:
                    assets[frame_id] = outputs.tolist()
            print(outputs.shape)
            team_boxes = [[] for _ in range(len(team_colors))]
            ball_boxes = []

            max_ball_output = None
            team_targets = []

            if args.track_before_knn:
                player_boxes = []
                for output in outputs:
                    if output[5] == 1:
                        player_boxes.append(output.tolist())
                    elif output[5] == 0:
                        if max_ball_output is None or output[4] > max_ball_output[4]:
                            max_ball_output = output
                player_targets = trackers.update(np.array(player_boxes), [img_info['height'], img_info['width']],
                               [img_info['height'], img_info['width']])
                for player_target in player_targets:
                    player_box = player_target.tlbr

                    team_id = team_assigner.get_player_team_test(frame, player_box, "")
                    team_boxes[team_id].append(player_target)
                team_targets = team_boxes

            else:
                for output in outputs:
                    if output[5] == 1:
                        team_id = team_assigner.get_player_team_test(frame, output[:4], "")
                        team_boxes[team_id].append(output.tolist())
                    elif output[5] == 0:
                        if max_ball_output is None or output[4] > max_ball_output[4]:
                            max_ball_output = output

                for boxes, tracker in zip(team_boxes, trackers):
                    team_target = tracker.update(np.array(boxes), [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                    team_targets.append(team_target)

            if max_ball_output is not None:
                ball_boxes.append(max_ball_output.tolist())

            img = frame

            for t_idx, team_target in enumerate(team_targets):
                foot_locations = []
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in team_target:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    foot_location = [tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3]]
                    foot_locations.append(foot_location)
                    real_foot_location = cv2.perspectiveTransform(np.array([[foot_location]]), matrix).tolist()[0][0]
                    if t_idx == 0:
                        team1_dict[tid].append(real_foot_location)
                    elif t_idx == 1:
                        team2_dict[tid].append(real_foot_location)
                    elif t_idx == 2:
                        goalkeeper_dict[tid].append(real_foot_location)
                    elif t_idx == 3:
                        #referee_dict[tid].append(real_foot_location)
                        referee_dict[tid].append([real_foot_location,frame_id])

                if len(foot_locations) == 0:
                    continue
                foot_locations = np.array([foot_locations])
                real_foot_locations = cv2.perspectiveTransform(foot_locations, matrix)
                real_foot_locations = real_foot_locations[0]
                for real_foot_location in real_foot_locations:
                    cv2.circle(top_view_img, (int(real_foot_location[0]), int(real_foot_location[1])), 20,
                               tuple(team_colors[t_idx].tolist()), -1)
                img = plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=0,
                                          color=tuple(team_colors[t_idx].tolist()))

            if args.no_ball_tracker:
                if max_ball_output is not None:
                    ball_box = ball_boxes[0][:4]
                    ball_box = [ball_box[0], ball_box[1], ball_box[2] - ball_box[0], ball_box[3] - ball_box[1]]
                else:
                    ball_box = []
            else:
                ball_targets = ball_tracker.update(np.array(ball_boxes), [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                print('ball_num',len(ball_targets),'ball_detct',len(ball_boxes))
                ball_box = ball_targets[0].tlwh if ball_targets else []

            if len(ball_box) > 0:
                ball_location = [ball_box[0] + ball_box[2] / 2, ball_box[1] + ball_box[3]]
                ball_locations = np.array([[ball_location]])
                real_ball_locations = cv2.perspectiveTransform(ball_locations, matrix)
                real_ball_locations = real_ball_locations[0][0]
                real_ball_history.append(real_ball_locations.tolist())
                cv2.circle(top_view_img, (int(real_ball_locations[0]), int(real_ball_locations[1])), 20,(0,255,0), -1)
                img = plot_tracking(img, [ball_box], [1], frame_id=frame_id + 1, fps=0,color=(0,255,0))

            analysis.process(team1_players=team1_dict,
                             team2_players=team2_dict,
                             side_referees=referee_dict,
                             goalkeepers=goalkeeper_dict,
                             balls=real_ball_history,
                             frame_id=frame_id,
                             matrix=matrix)
            analysis.visualize(img)
            # top_view.process()
            top_view_img = cv2.resize(top_view_img, (tv_h, tv_w))
            cv2.imshow('Image', img)
            cv2.imshow('Top View', top_view_img)
            vid_writer.write(img)
            topview_writer.write(top_view_img)
            ch = cv2.waitKey(1)
            frame_id += 1
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

    if args.save_asset:
        with open(asset_path, 'w') as f:
            json.dump(assets, f, indent=4)

if __name__ == '__main__':
    args = make_parser().parse_args()
    predictor = Predictor(args)
    imageflow_demo(predictor, args)
