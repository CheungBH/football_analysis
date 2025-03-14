import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import defaultdict
import argparse
import copy
from team_assigner import TeamAssigner
import json
import cv2
import numpy as np
from analyser.analysis import AnalysisManager
import onnxruntime

from visualize import plot_tracking
from tracker import BYTETracker

team_colors = []
# team_box_colors = [(0, 0, 255), (0, 255, 0)]

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="../assets/best.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--court_image",
        type=str,
        default='court_reference.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='../assets/output_10.mp4',
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
        default=0.1,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.3,
        help="NMS threshould.",
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

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def preprocess(image, input_size, swap=(2, 0, 1)):
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
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

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
        self.session = onnxruntime.InferenceSession(args.model)
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def inference(self, ori_img):


        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        img, ratio = preprocess(ori_img, self.input_shape)
        img_info["ratio"] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}

        output = self.session.run(None, ort_inputs)
        if len(output) > 1:
            output = [output[0]]

        predictions = np.squeeze(output).T

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        return dets[:, :-1], img_info


def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    json_path =  ".".join(args.video_path.split(".")[:-1]) + '.json'
    assert os.path.exists(json_path), f"json file not found in {json_path}"

    vid_writer = cv2.VideoWriter(
        args.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    with open(json_path, 'r') as f:
        assets = json.load(f)
    court_points = np.array(assets["court_point"])
    game_points = np.array(assets["game_point"])
    matrix = np.array(assets["court_matrix"])
    team_colors = [np.array(color) for color in assets["team_colors"]]

    frame_id = 0
    team_assigner = TeamAssigner()


    top_view_img = cv2.imread(args.court_image)
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id == 0:
                team_assigner.assign_color(team_colors)
                team_box_colors = team_assigner.team_colors
                trackers = [BYTETracker(args, frame_rate=30) for _ in range(len(team_box_colors))]
                matrix, _ = cv2.findHomography(game_points, court_points, cv2.RANSAC)

            # outputs, img_info = predictor.inference(frame)
            outputs = np.array(assets[str(frame_id)])
            img_info = {"height": height, "width": width, "raw_img": frame}
            team_boxes = [[] for _ in range(len(team_box_colors))]

            for output in outputs:
                team_id = team_assigner.get_player_team_test(frame, output[:4])
                team_boxes[team_id].append(output)

            team_targets = []
            for boxes, tracker in zip(team_boxes, trackers):
                team_target = tracker.update(np.array(boxes), [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                team_targets.append(team_target)

            img = img_info['raw_img']
            team_bw_dict = defaultdict(dict)
            for t_idx, team_target in enumerate(team_targets):
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in team_target:
                    tlwh = t.tlwh
                    tid = t.track_id

                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    team_bw_dict[t_idx][tid] = [tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3]]

                img = plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=0,
                                          color=team_box_colors[t_idx])


            # top_view.process()
            cv2.imshow('Image', img)
            cv2.imshow('Top View', cv2.resize(top_view_img, (400, 800)))

            vid_writer.write(img)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1



if __name__ == '__main__':
    args = make_parser().parse_args()

    predictor = Predictor(args)
    imageflow_demo(predictor, args)