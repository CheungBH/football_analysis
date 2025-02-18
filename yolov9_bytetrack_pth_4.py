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
from analyser.preprocess import sync_frame
import json
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_boxes
from threading import Thread
from visualize import plot_tracking
from tracker import BYTETracker
import config.config as config
import math
from functools import reduce


team_colors = []
img_full_list = []

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=r"D:\tmp\2.7\best.pt",
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
        default=r'D:\tmp\1.3\2',
        help="Path to your input video folder",
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
        "--start_frames",
        default=['0','0','0','0'],
        nargs='+',
        help='mask'
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
        return output.detach().cpu(), img_info
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

    def batch_inference(self, imgs):
        imgs_info = []
        input_imgs = []
        for img in imgs:
            height, width = img.shape[:2]
            img_info = {}
            img_info["height"] = height
            img_info["width"] = width
            img_info["raw_img"] = img
            im = letterbox(img, list(self.input_shape), stride=self.stride, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            img_info["tensor_size"] = im.shape
            input_imgs.append(im[None])
            imgs_info.append(img_info)
        batch_imgs = torch.ones(1, 3, input_imgs[0].shape[2], input_imgs[0].shape[3]).to(self.model.device)
        for input_img in input_imgs:
            batch_imgs = torch.cat((batch_imgs, input_img), dim=0)
        # if len(im.shape) == 3:
        #     im = im[None]  # expand for batch dim
        preds = self.model(batch_imgs[1:,...])
        preds = non_max_suppression(preds, self.args.score_thr, self.args.nms_thr, None, False, max_det=1000)
        # pred = pred[0]
        if preds is None:
            return None, img_info
        outputs = []
        for idx, (pred, img_info) in enumerate(zip(preds, imgs_info)):
            output = pred#[0]
            height, width, raw_img = img_info["height"], img_info["width"], img_info["raw_img"]
            tensor_size = img_info["tensor_size"]
            output[:, :4] = scale_boxes(tensor_size[1:], output[:, :4], raw_img.shape).round()
            outputs.append(output.detach().cpu())
        return outputs, imgs_info



def imageflow_demo(predictor, args):
    video_names = config.cam4_videos_name
    video_folder = args.video_path
    # files_in_folder = os.listdir(video_folder)
    # mp4_files = [file for file in files_in_folder if file.endswith('.mp4')]
    # video_paths = []
    # for mp4 in mp4_files:
    #     video_path = os.path.join(video_folder,mp4)
    #     video_paths.append(video_path)1
    video_paths = [os.path.join(video_folder, name) for name in video_names]
    #sync_frame.resample_videos(video_paths, 30)
    start_frames = args.start_frames if args.start_frames else []
    if start_frames == []:
        for video_path in video_paths:
            start_frame = sync_frame.select_start_frame(video_path)
            start_frames.append(start_frame)
    print(start_frames)
    caps = [cv2.VideoCapture(path) for path in video_paths]
    width_list,height_list,fps_list = [],[],[]
    for cap, start_frame in zip(caps, start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        width_list.append(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_list.append(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_list.append(round(cap.get(cv2.CAP_PROP_FPS)))


    width = caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)  # float,
    height = caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    fpsmin = reduce(math.gcd,fps_list)
    if args.use_json:
        args.save_asset = False
    tv_h, tv_w = config.topview_height, config.topview_width
    real_h, real_w = config.real_video_height, config.real_video_width
    vid_writer = cv2.VideoWriter(
        args.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (real_w, real_h)
    )
    topview_writer = cv2.VideoWriter(
        "/".join(args.output_video_path.split("/")[:-1]) + "top_view.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (tv_w, tv_h)
    )
    frame_id = 0
    team_assigner = TeamAssigner()
    points = []
    img_list = []
    def click_court(court_img):
        points.clear()
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(court_img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('click_court', court_img)
                # if len(points) > 4:
                #     points.pop(0)
                print(points)

        height, width, channel = court_img.shape
        cv2.namedWindow("click_court", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("click_court", width, height)
        cv2.setMouseCallback("click_court", click_event)

        while True:
            cv2.imshow("click_court", court_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to break
                break
            court_img = copy.deepcopy(court_img)

        print(points)
        cv2.destroyAllWindows()


    # real_player_location = {defaultdict}
    top_view_img_tpl = cv2.imread(args.court_image)
    real_ball_history=[]
    team1_dict = defaultdict(list)
    team2_dict = defaultdict(list)
    goalkeeper1_dict = defaultdict(list)
    goalkeeper2_dict = defaultdict(list)
    referee_dict = defaultdict(list)
    analysis = AnalysisManager(config.check_action, ((0, 0)))


    while True:
        if frame_id == 10:
            break
        top_view_img = copy.deepcopy(top_view_img_tpl)
        ret_vals,frames_list=[],[]
        for i,cap in enumerate(caps):
            frame_interval = round(cap.get(cv2.CAP_PROP_FPS))/fpsmin
            if frame_id !=0 :
                for i in range(int(frame_interval)):
                    ret_val,frame = cap.read()
            else:
                ret_val, frame = cap.read()
            ret_vals.append(ret_val)
            frames_list.append(frame)
        assets = [[] for _ in range(len(ret_vals))]
        if sum(ret_vals) == len(ret_vals):
            if frame_id == 0:
                if args.use_json:
                    # os.path.dirname()
                    # os.path.basename()
                    # json_name = os.path.join()
                    json_path = os.path.join(args.video_path, "assets.json")

                    with open(json_path, 'r') as f:
                        assets = json.load(f)
                    matrix_list=[]
                    for i in assets:
                        court_points = np.array(i["court_point"])
                        game_points = np.array(i["game_point"])
                        matrix, _ = cv2.findHomography(game_points, court_points, cv2.RANSAC)
                        matrix_list.append(matrix)
                    #matrix = np.array(assets["court_matrix"])
                    # team_colors = {0: np.array([0, 0, 255], dtype=np.uint8),
                    #                1: np.array([125, 125, 125], dtype=np.uint8),
                    #                2: np.array([255, 0, 0], dtype=np.uint8), 3: np.array([0, 0, 0], dtype=np.uint8)}
                    # team_assigner.assign_color(team_colors)


                else:
                    # team_assigner.assign_color(team_colors)
                    # team_colors = {0: np.array([0, 0, 255], dtype=np.uint8),
                    #                1: np.array([125, 125, 125], dtype=np.uint8),
                    #                2: np.array([255, 0, 0], dtype=np.uint8), 3: np.array([0, 0, 0], dtype=np.uint8)}

                    court_img = cv2.imread(args.court_image)
                    matrix_list= []
                    for idx,frame in enumerate(frames_list):
                        real_court_img = copy.deepcopy(frame)
                        points=[]
                        click_court(real_court_img)
                        game_points = np.array(points)
                        time.sleep(1)
                        points=[]
                        click_court(court_img)
                        court_points = np.array(points)
                        matrix, _ = cv2.findHomography(game_points, court_points, cv2.RANSAC)
                        matrix_list.append(matrix)
                        if args.save_asset:
                            asset_name = os.path.join(args.video_path, 'assets.json')
                            asset_path = os.path.join(args.video_path,asset_name)
                            assets[idx] = {
                                "view": f"view{idx}",
                                "court_matrix": matrix.tolist(),
                                "game_point": game_points.tolist(),
                                "court_point": court_points.tolist(),
                                # "team_colors": [color.tolist() for index, color in team_colors.items()],
                            }
                    print(matrix_list)
                    cv2.destroyAllWindows()
                # if args.use_color:

                # color_json_name = os.path.dirname(args.video_path)+ 'color.json'
                color_json_path = os.path.join(args.video_path, 'color.json')
                with open(color_json_path, 'r') as f:
                    color_asset = json.load(f)

                team_colors = color_asset
                team_colors = {int(k): v for k, v in team_colors.items()}
                team_assigner.assign_color(team_colors)
                # team_colors = team_assigner.team_colors
                # team_box_colors = team_colors
                print(team_colors)


                if args.track_before_knn:
                    tracker_list = [BYTETracker(args, frame_rate=30),
                                BYTETracker(args, frame_rate=30),
                                BYTETracker(args, frame_rate=30),
                                BYTETracker(args, frame_rate=30)]
                else:
                    tracker_list = [[BYTETracker(args, frame_rate=30) for _ in range(len(team_colors))],
                                [BYTETracker(args, frame_rate=30) for _ in range(len(team_colors))],
                                [BYTETracker(args, frame_rate=30) for _ in range(len(team_colors))],
                                [BYTETracker(args, frame_rate=30) for _ in range(len(team_colors))]
                                ]

                ball_tracker = BYTETracker(args, frame_rate=30)

            yolo_outputs, imgs_info = predictor.batch_inference(frames_list)
            for index, (frame, outputs, img_info) in enumerate(zip(frames_list, yolo_outputs, imgs_info)):
            # for index,frame in enumerate(frames_list):
                #trackers = [BYTETracker(args, frame_rate=30) for _ in range(len(team_colors))]
                #trackers = trackers
                # ai = predictor.batch_inference([frame, frame, frame, frame])

                # outputs, img_info = predictor.inference(frame)
                # outputs, img_info = yolo_output
                matrix = matrix_list[index]
                trackers = tracker_list[index]

                print(outputs.shape)
                team_boxes = [[] for _ in range(len(team_colors))]
                ball_boxes = []

                max_ball_output = None
                team_targets = [[],[],[],[],[]]

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
                        try:
                            team_id = team_assigner.get_player_team_test(frame, player_box, "",team_colors)
                        except:
                            team_id = 0
                        team_boxes[team_id].append(player_target)
                    team_targets[index] = team_boxes

                else:
                    for output in outputs:
                        if output[5] == 1:
                            team_id = team_assigner.get_player_team_test(frame, output[:4], "",team_colors)
                            team_boxes[team_id].append(output.tolist())
                        elif output[5] == 0:
                            if max_ball_output is None or output[4] > max_ball_output[4]:
                                max_ball_output = output

                    for boxes, tracker in zip(team_boxes, trackers):
                        team_target = tracker.update(np.array(boxes), [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                        team_targets[index].append(team_target)

                if max_ball_output is not None:
                    ball_boxes.append(max_ball_output.tolist())

                img = frame

                for t_idx, team_target in enumerate(team_targets[index]):
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
                            goalkeeper1_dict[tid].append(real_foot_location)
                        elif t_idx == 3:
                            goalkeeper2_dict[tid].append(real_foot_location)
                        elif t_idx == 4:
                            #referee_dict[tid].append(real_foot_location)
                            referee_dict[tid].append([real_foot_location,frame_id])

                    if len(foot_locations) == 0:
                        continue
                    foot_locations = np.array([foot_locations])
                    real_foot_locations = cv2.perspectiveTransform(foot_locations, matrix)
                    real_foot_locations = real_foot_locations[0]
                    t_color = team_colors[t_idx]
                    t_color = t_color if isinstance(t_color, list) else t_color.tolist()
                    for real_foot_location in real_foot_locations:
                        cv2.circle(top_view_img, (int(real_foot_location[0]), int(real_foot_location[1])), 20, tuple(t_color), -1)
                    img = plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=0,  color=t_color)

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
                    ball_location = [ball_box[0] + ball_box[2] / 2, ball_box[1] + ball_box[3]/2]
                    ball_locations = np.array([[ball_location]])
                    real_ball_locations = cv2.perspectiveTransform(ball_locations, matrix)
                    real_ball_locations = real_ball_locations[0][0]
                    real_ball_history.append(real_ball_locations.tolist())
                    cv2.circle(top_view_img, (int(real_ball_locations[0]), int(real_ball_locations[1])), 20,(0,255,0), -1)
                    img = plot_tracking(img, [ball_box], [1], frame_id=frame_id + 1, fps=0,color=(0,255,0))
                resized_frame = cv2.resize(img, (real_w//2, real_h//2))
                img_list.append(resized_frame)

            analysis.process(team1_players=team1_dict,
                             team2_players=team2_dict,
                             side_referees=referee_dict,
                             goalkeepers1=goalkeeper1_dict,
                             goalkeepers2=goalkeeper2_dict,
                             balls=real_ball_history,
                             frame_id=frame_id)
            analysis.visualize(img)
            flag = analysis.flag
            # top_view.process()
            top_view_img = cv2.resize(top_view_img, (tv_w, tv_h))
            #cv2.imshow('Image', img)

            if len(img_list) == 4:
                top_row = np.hstack(img_list[:2])
                bottom_row = np.hstack(img_list[2:])
                combined_frame = np.vstack([top_row, bottom_row])
                # Display the combined frame
                cv2.imshow("Combined Frame", combined_frame)
                cv2.imshow('Top View', top_view_img)
                img_full_list.append(img_list)
                img_list = []
                if flag != 0:
                    start_index = max(0, frame_id - 30)
                    end_index = frame_id
                    output_video_path = f'result/output_segment_{index}_{frame_id}.mp4'
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (540, 360))
                    # Write frames to the video
                    for i in range(start_index, end_index):
                        out.write(img_full_list[i][index])

                    # Release the video writer
                    out.release()
                frame_id += 1
            vid_writer.write(cv2.resize(combined_frame, (real_w, real_h)))
            topview_writer.write(top_view_img)
            ch = cv2.waitKey(1)

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
