from analyser.utils import FrameQueue
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
from analyser.topview import TopViewGenerator
from team_assigner_dinov2 import TeamAssigner
from team_assigner import TeamAssigner as TeamAssignerKNN
from analyser.flag_manager import FlagManager

from analyser.analysis import AnalysisManager
from analyser.preprocess import sync_frame
import json
import cv2
from analyser.knn_try.get_jersey_color import process_images
from utils.general import non_max_suppression, scale_boxes, scale_and_remove_boxes
from utils.crop_img_sliding_window import sliding_window_crop
from visualize import plot_tracking
from tracker import BYTETracker
import config.config as config
import math
from functools import reduce


team_colors = []
img_full_list = []
map_cam_idx = [0,1,2,3]


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference")
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=r"assets/checkpoints/best.pt",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Path to your input image.",
    )
    parser.add_argument(

        "--jersey_folder",
        type=str,
        default='',
        help="Path to your input folder of jersey.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        required=True,
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
        default="",
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
        default=0.45,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=640,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=320,
        help="Score threshould to filter the result.",
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
        "--save_cropped_humans",
        default="",
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
        help="Load json for court",
    )
    parser.add_argument(
        "--show_video",
        action="store_true",
        help="Load json for court",
    )
    parser.add_argument(
        "--use_saved_box",
        action="store_true",
        help="Load box json for fast inference",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--stop_at", type=int, default=-1, help="which frame to stop")
    parser.add_argument("--start_with", type=int, default=-1, help="which frame to start")

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


import numpy as np


def post_nms(boxes, iou_threshold=0.8):
    # return boxes

    if boxes.numel() == 0:
        return torch.empty((0, 6), device=boxes.device)  # Return empty tensor if no boxes

    # Sort boxes by confidence score in descending order
    boxes = boxes[torch.argsort(boxes[:, 4], descending=True)]

    selected_boxes = []
    while boxes.size(0) > 0:
        chosen_box = boxes[0]  # Select the highest confidence box
        selected_boxes.append(chosen_box)

        if boxes.size(0) == 1:
            break  # No more boxes left to compare

        other_boxes = boxes[1:]

        # Compute IoU (Intersection over Union)
        x1 = torch.maximum(chosen_box[0], other_boxes[:, 0])
        y1 = torch.maximum(chosen_box[1], other_boxes[:, 1])
        x2 = torch.minimum(chosen_box[2], other_boxes[:, 2])
        y2 = torch.minimum(chosen_box[3], other_boxes[:, 3])

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        chosen_area = (chosen_box[2] - chosen_box[0]) * (chosen_box[3] - chosen_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        iou = intersection / (chosen_area + other_areas - intersection)

        # Keep boxes below the IoU threshold OR different class
        boxes = other_boxes[(iou < iou_threshold) | (chosen_box[5] != other_boxes[:, 5])]

    return torch.stack(selected_boxes)

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

    def batch_inference_cropped(self, imgs_dict, frame):
        imgs = imgs_dict["images"]
        img_tl = imgs_dict["top_left"]

        for idx, img in enumerate(imgs):
            cv2.imwrite("img/{}.jpg".format(idx), img)

        height, width = frame.shape[:2]
        img_raw_info = {}
        img_raw_info["height"] = height
        img_raw_info["width"] = width
        img_raw_info["raw_img"] = frame


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
        preds = non_max_suppression(preds[0][0], self.args.score_thr, self.args.nms_thr, None, False, max_det=1000)
        # pred = pred[0]
        if preds is None:
            return None, img_raw_info
        outputs = []

        final_preds = torch.zeros(1, 6)
        for idx, (pred, img_info) in enumerate(zip(preds, imgs_info)):
            output = pred#[0]
            height, width, raw_img = img_info["height"], img_info["width"], img_info["raw_img"]
            tensor_size = img_info["tensor_size"]
            output = scale_and_remove_boxes(tensor_size[1:], output, raw_img.shape)#.round()
            # im_tl = torch.Size(img_tl[idx])

            output[:,0] += img_tl[idx][0]
            output[:,2] += img_tl[idx][0]
            output[:,1] += img_tl[idx][1]
            output[:,3] += img_tl[idx][1]
            outputs.append(output.detach().cpu())
        for output in outputs:
            final_preds = torch.cat((final_preds, output), dim=0)
        return post_nms(final_preds[1:]), img_raw_info


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
    #     video_paths.append(video_path)
    whole_analysis_criterion = ["ball_out_range", "delay_restart"]
    all_actions = config.check_action
    single_action = [action for action in all_actions if action not in whole_analysis_criterion]
    whole_action = [action for action in all_actions if action in whole_analysis_criterion]


    args.track_before_knn = True
    args.no_ball_tracker = True
    args.save_tmp_tv = "tv"
    # args.use_json = True
    # args.show_video = False
    court_image = os.path.join(args.video_path, "court.jpg") if not args.court_image else args.court_image


    if args.jersey_folder:
        output_folder = args.jersey_folder + "_output"
        os.makedirs(output_folder, exist_ok=True)
        output_json = os.path.join(args.video_path, "color.json")
        process_images(args.jersey_folder, output_folder, output_json)
    else:
        print("No jersey folder provided. Make sure to provide the color.json")

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

    if args.save_cropped_humans:
        os.makedirs(args.save_cropped_humans, exist_ok=True)

    # fps = caps[0].get(cv2.CAP_PROP_FPS)
    fpsmin = reduce(math.gcd,fps_list)
    frame_queue = fpsmin * 5
    # if args.use_json:
    #     args.save_asset = False
    tv_h, tv_w = config.topview_height, config.topview_width
    real_h, real_w = config.real_video_height, config.real_video_width
    vid_writer = cv2.VideoWriter(
        args.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fpsmin, (real_w, real_h)
    )
    print("Save video to: ", args.output_video_path)
    print(os.path.exists(args.output_video_path))
    tv_path = os.path.join(os.path.dirname(args.output_video_path), "top_view.mp4")
    topview_writer = cv2.VideoWriter(tv_path, cv2.VideoWriter_fourcc(*"mp4v"), fpsmin, (tv_w, tv_h)
    )

    if args.save_tmp_tv:
        tmp_tv_path = os.path.join(os.path.dirname(args.output_video_path), "top_view_single.mp4")
        tmp_tv_writer = cv2.VideoWriter(tmp_tv_path, cv2.VideoWriter_fourcc(*"mp4v"), fpsmin, (int(tv_w*2), int(tv_h*2))
        )

    frame_id = 0

    team_assigner = TeamAssigner(root_folder=r"assets/dino/global_features", model_path="assets/dino/model.pth")
    team_assigner_knn = TeamAssignerKNN()

    points = []

    def click_court(court_img):
        points.clear()
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(court_img, (x, y), 5, (0, 0, 255), -1)
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
            if key == 27:  # 0SC key to break
                break
            court_img = copy.deepcopy(court_img)

        print(points)
        cv2.destroyAllWindows()


    top_view_img_tpl = cv2.imread(court_image)
    all_player_dict = [defaultdict(list)for _ in range(4)]
    filtered_dict = [defaultdict(list)for _ in range(4)]
    team1_dict = [defaultdict(list)for _ in range(4)]
    team2_dict = [defaultdict(list)for _ in range(4)]
    goalkeeper1_dict = [defaultdict(list)for _ in range(4)]
    goalkeeper2_dict = [defaultdict(list)for _ in range(4)]
    referee_dict = [defaultdict(list)for _ in range(4)]
    analysis_list = [AnalysisManager(single_action, ((0, 0))) for _ in range(4)]
    analysis_wholegame = AnalysisManager(whole_action, ((0, 0)), display_x=500)

    # analysis = AnalysisManager(config.check_action, ((0, 0)))
    frames_queue_ls = [FrameQueue(frame_queue) for _ in range(len(caps))]
    topview_queue = FrameQueue(frame_queue)
    merged_list = []

    PlayerTopView = TopViewGenerator((50,50,1100,720))
    flag_manager = FlagManager(config.check_action, frame_duration=fpsmin*60, min_activate_flag=fpsmin*5)

    if args.use_saved_box:
        box_asset_path = os.path.join(args.video_path, 'yolo.json')
        assert os.path.exists(box_asset_path), "The box asset file does not exist."
        with open(box_asset_path, 'r') as f:
            box_assets = json.load(f)
    else:
        if args.save_asset:
            box_asset_path = os.path.join(args.video_path, 'yolo.json')
            box_assets = {}
            # if os.path.exists(box_asset_path):
            #     input("The box asset file already exists, do you want to overwrite it? Press Enter to continue, or Ctrl+C to exit.")
            box_f = open(box_asset_path, 'w')

    while True:

        if frame_id == args.stop_at:
            break

        # try:
        img_list = []
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

        if frame_id < args.start_with and frame_id != 0:
            frame_id += 1
            if frame_id % 100 == 0:
                print(frame_id)
            continue

        if sum(ret_vals) == len(ret_vals):
            frames_list = [cv2.resize(frame, (real_w, real_h)) for frame in frames_list]
            flag_list=[]

            if frame_id == 0:
                if args.use_json:
                    # json_path = '/media/hkuit164/Backup/football_analysis/datasets/assets.json'
                    json_path = os.path.join(args.video_path, "assets.json")

                    with open(json_path, 'r') as f:
                        assets = json.load(f)
                    matrix_list=[[] for _ in range(4)]
                    for idx,i in enumerate(assets):
                        court_points = np.array(i["court_point"])
                        game_points = np.array(i["game_point"])
                        matrix, _ = cv2.findHomography(game_points, court_points, cv2.RANSAC)
                        matrix_list[idx].append(matrix)

                else:
                    court_img = cv2.imread(court_image)
                    matrix_list= [[] for _ in range(4)]
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
                        matrix_list[idx].append(matrix)
                        if args.save_asset:
                            asset_name = os.path.join(args.video_path, 'assets.json')
                            asset_path = os.path.join(args.video_path,asset_name)
                            assets[idx] = {
                                "view": f"view{idx}",
                                "court_matrix": matrix.tolist(),
                                "game_point": game_points.tolist(),
                                "court_point": court_points.tolist(),
                            }
                    print(matrix_list)
                    cv2.destroyAllWindows()

                color_json_path = os.path.join(args.video_path, 'color.json')
                with open(color_json_path, 'r') as f:
                    color_asset = json.load(f)
                team_colors = color_asset
                team_colors = {int(k): v for k, v in team_colors.items()}
                team_assigner.assign_color(team_colors)
                print(team_colors)

                if args.track_before_knn:
                    tracker_list = [BYTETracker(args, frame_rate=fpsmin),
                                BYTETracker(args, frame_rate=fpsmin),
                                BYTETracker(args, frame_rate=fpsmin),
                                BYTETracker(args, frame_rate=fpsmin)]
                else:
                    tracker_list = [[BYTETracker(args, frame_rate=fpsmin) for _ in range(len(team_colors))],
                                [BYTETracker(args, frame_rate=fpsmin) for _ in range(len(team_colors))],
                                [BYTETracker(args, frame_rate=fpsmin) for _ in range(len(team_colors))],
                                [BYTETracker(args, frame_rate=fpsmin) for _ in range(len(team_colors))]
                                ]

                ball_tracker = BYTETracker(args, frame_rate=fpsmin)

            cropped_frames = {}
            for index,frame in enumerate(frames_list):
                cropped_frames[index] = sliding_window_crop(frame, (args.crop_size, args.crop_size), (args.window_size, args.window_size))

            all_players, all_balls = [], []
            if not args.use_saved_box:
                yolo_outputs, imgs_info = [], []
                for index in range(len(frames_list)):
                    yolo_output, img_info = predictor.batch_inference_cropped(cropped_frames[index], frames_list[index])
                    yolo_outputs.append(yolo_output)
                    imgs_info.append(img_info)

                if args.save_asset:
                    box_assets[frame_id] = {}
                    for yolo_idx, yolo_output in enumerate(yolo_outputs):
                        box_assets[frame_id][yolo_idx] = yolo_output.tolist()

            else:
                yolo_outputs = []
                for index in range(len(frames_list)):
                    yolo_output = torch.tensor(box_assets[str(frame_id)][str(index)])
                    yolo_outputs.append(yolo_output)
                imgs_info = []
                for frame in frames_list:
                    img_info = {}
                    height, width = frame.shape[:2]
                    img_info["height"] = height
                    img_info["width"] = width
                    img_info["raw_img"] = frame
                    imgs_info.append(img_info)

            real_ball_locations_all = []
            # yolo_outputs, imgs_info = predictor.batch_inference(frames_list)
            for index, (frame, outputs, img_info) in enumerate(zip(frames_list, yolo_outputs, imgs_info)):
                frames_queue_ls[index].push_frame(frame)
                real_ball_history=[]

                matrix = matrix_list[index][0]
                trackers = tracker_list[index]

                # print(outputs.shape)
                team_boxes = [[] for _ in range(len(team_colors))]
                team_boxes_whole = []
                players_real_location = [defaultdict(list)for _ in range(4)]
                ball_boxes = []

                max_ball_output = None
                all_ball_output = []
                team_targets = [[],[],[],[]]
                # team_targets_add = [[],[],[],[]]

                if args.save_cropped_humans:
                    for idx, output in enumerate(outputs):
                        if output[5] == 1:
                            box = output[:4]
                            cropped_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            cv2.imwrite(f"{args.save_cropped_humans}/human_{frame_id}_{index}_{idx}.jpg", cropped_img)

                if args.track_before_knn:
                    player_boxes = []
                    for output in outputs:
                        if output[5] == 1:
                            # Height > Width
                            height, width = output[3] - output[1], output[2] - output[0]
                            if height > width and height / width < 3:
                                player_boxes.append(output.tolist())

                        elif output[5] == 0:
                            all_ball_output.append(output.tolist())
                            if max_ball_output is None or output[4] > max_ball_output[4]:
                                max_ball_output = output
                    player_targets = trackers.update(np.array(player_boxes), [img_info['height'], img_info['width']],
                                   [img_info['height'], img_info['width']])

                    team_ids =  team_assigner.get_player_whole_team(frame, [target.tlbr for target in player_targets],
                                                                    index, team_colors=team_colors, cam_idx=index)
                    team_ids_knn = team_assigner_knn.get_player_whole_team(frame, [target.tlbr for target in player_targets],
                                                                    index, team_colors=team_colors, cam_idx=index)
                    t_final_id = []
                    for t_id_dino, t_id_knn in zip(team_ids, team_ids_knn):
                        if t_id_knn != 4 and t_id_dino == 4:
                            t_final_id.append(t_id_knn)
                        else:
                            t_final_id.append(t_id_dino)
                    for player_target, team_id in zip(player_targets, team_ids):
                        team_boxes[team_id].append(player_target)
                        team_boxes_whole.append(player_target)
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
                real_foot_locations = [[] for _ in range(4)]
                for t_idx, team_target in enumerate(team_targets[index]):
                    foot_locations = [[] for _ in range(4)]

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
                        foot_locations[index].append(foot_location)
                        real_foot_location = cv2.perspectiveTransform(np.array([[foot_location]]), matrix).tolist()[0][0]
                        players_real_location[index][tid] = real_foot_location
                        real_foot_locations[index].append(real_foot_location + [t_idx, team_colors[t_idx]])
                        all_player_dict[index][tid].append(real_foot_location)

                    # filtered_dict[index] = {key: all_player_dict[index][key] for key in online_ids}

                        if t_idx == 0:
                            team1_dict[index][tid].append(real_foot_location)
                        elif t_idx == 1:
                            team2_dict[index][tid].append(real_foot_location)
                        elif t_idx == 2:
                            goalkeeper1_dict[index][tid].append(real_foot_location)
                        elif t_idx == 3:
                            goalkeeper2_dict[index][tid].append(real_foot_location)
                        elif t_idx == 4:
                            #referee_dict[tid].append(real_foot_location)
                            referee_dict[index][tid].append([real_foot_location,frame_id])


                    if len(foot_locations) == 0:
                        continue
                    # foot_locations = np.array([foot_locations])
                    # real_foot_locations = cv2.perspectiveTransform(foot_locations, matrix)
                    # real_foot_locations = real_foot_locations[0]
                    # t_color =
                    # t_color = t_color if isinstance(t_color, list) else t_color.tolist()
                    all_players += real_foot_locations[index]
                    # for real_foot_location in real_foot_locations[index]:
                    #     all_players.append(real_foot_location + [t_idx, team_colors[t_idx]])

                    #     cv2.circle(top_view_img, (int(real_foot_location[0]), int(real_foot_location[1])), 20, tuple(t_color), -1)
                    img = plot_tracking(img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=0,  color=team_colors[t_idx])

                if args.no_ball_tracker:
                    all_ball_boxes = []
                    if max_ball_output is not None:
                        ball_box = ball_boxes[0][:4]
                        ball_box = [ball_box[0], ball_box[1], ball_box[2] - ball_box[0], ball_box[3] - ball_box[1]]
                        for single_ball_box in all_ball_output:
                            single_ball = single_ball_box[:4]
                            all_ball_boxes.append([single_ball[0], single_ball[1], single_ball[2] - single_ball[0],
                                                   single_ball[3] - single_ball[1]])
                    else:
                        ball_box = []
                else:
                    ball_targets = ball_tracker.update(np.array(ball_boxes), [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                    print('ball_num',len(ball_targets),'ball_detct',len(ball_boxes))
                    ball_box = ball_targets[0].tlwh if ball_targets else []



                real_ball_locations_singe_cam = []
                if len(all_ball_boxes) > 0:
                    for boxes in all_ball_boxes:
                        ball_location = [boxes[0] + boxes[2] / 2, boxes[1] + boxes[3]/2]
                        ball_locations = np.array([[ball_location]])
                        real_ball_locations = cv2.perspectiveTransform(ball_locations, matrix)
                        real_ball_locations = real_ball_locations[0][0].tolist()
                        real_ball_locations_singe_cam.append(real_ball_locations)
                        real_ball_locations_all.append(real_ball_locations)
                        # all_balls.append(real_ball_locations)
                        cv2.circle(top_view_img, (int(real_ball_locations[0]), int(real_ball_locations[1])), 20,(0,255,0), -1)
                        # img = plot_tracking(img, [ball_box], [1], frame_id=frame_id + 1, fps=0,color=(0,255,0))
                real_ball_locations=[]
                if len(ball_box) > 0:
                    ball_location = [ball_box[0] + ball_box[2] / 2, ball_box[1] + ball_box[3]/2]
                    ball_locations = np.array([[ball_location]])
                    real_ball_locations = cv2.perspectiveTransform(ball_locations, matrix)
                    real_ball_locations = real_ball_locations[0][0].tolist()
                    real_ball_history.append(real_ball_locations)
                    all_balls.append(real_ball_locations)
                    # cv2.circle(top_view_img, (int(real_ball_locations[0]), int(real_ball_locations[1])), 20,(0,255,0), -1)
                    img = plot_tracking(img, [ball_box], [1], frame_id=frame_id + 1, fps=0,color=(0,255,0))

                resized_frame = cv2.resize(img, (real_w//2, real_h//2))
                img_list.append(resized_frame)
                # analysis_list[index].process(team1_players=team1_dict[index],
                #                  team2_players=team2_dict[index],
                #                  side_referees=referee_dict[index],
                #                  goalkeepers1=goalkeeper1_dict[index],
                #                  goalkeepers2=goalkeeper2_dict[index],
                #                  balls=real_ball_history,
                #                  frame_id=frame_id,
                #                  matrix=matrix,
                #                 frame_queue=frame_queue)


                analysis_list[index].process(players = players_real_location[index], balls=real_ball_locations_singe_cam,
                    frame_id=frame_id,matrix=matrix,frame_queue=frame_queue)
                analysis_list[index].visualize(img_list[index])

                # if index ==3:


                # for i in range(len(real_foot_locations[index])):
                #     cv2.circle(top_view_img, (int(real_foot_locations[index][i][0]), int(real_foot_locations[index][i][1])), 20, (0, 255, 0), -1)
                flag_list.append(analysis_list[index].flag_dict)
                if args.save_tmp_tv:
                    PlayerTopView.save_topview_img(top_view_img=copy.deepcopy(top_view_img_tpl),
                                                   players=real_foot_locations[index],
                                                   balls=real_ball_locations_singe_cam, frame_idx=index, path=args.save_tmp_tv)
                    cv2.imwrite(f"{args.save_tmp_tv}/raw_{index}.jpg", img)
            if args.save_tmp_tv:
                PlayerTopView.save_topview_img(copy.deepcopy(top_view_img_tpl), all_players, all_balls, "whole", args.save_tmp_tv)
                single_tv_frame = PlayerTopView.save_tmp_videos(args.save_tmp_tv, tmp_tv_writer, size=(tv_w, tv_h))
                cv2.imshow("Single Top View", single_tv_frame)
            PlayerTopView.process(all_players, all_balls)
            PlayerTopView.visualize(top_view_img)
            cv2.putText(top_view_img, f"Frame: {frame_id}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            analysis_wholegame.process(balls=real_ball_locations_all, players = players_real_location[index],
                    frame_id=frame_id,matrix=matrix,frame_queue=frame_queue)
            analysis_wholegame.visualize(img_list[1])

            if args.save_tmp_tv:
                cv2.imwrite(f"{args.save_tmp_tv}/tv_whole.jpg", top_view_img)

            # merged_dict = {}
            # for lst in flag_list:
            #     for key, value in lst:
            #         if key not in merged_dict:
            #             merged_dict[key] = value
            #         else:
            #             merged_dict[key] = merged_dict[key] or value
            # merged_value = sum(merged_dict.values())
            # merge_list = [[key, value] for key, value in merged_dict.items()]
            # merged_list.append(merge_list)
            # merged_list[-1].append(frame_id)
            #
            # analysis_file = os.path.join(args.video_path, "analysis.txt")
            # with open(analysis_file, 'w') as f:
            #     for item in merged_list:
            #         f.write(str(item) + '\n')

            # top_view.process()
            top_view_img = cv2.resize(top_view_img, (tv_w, tv_h))
            topview_queue.push_frame(top_view_img)
            #cv2.imshow('Image', img)

            flag_manager.update(analysis_wholegame.flag_dict, flag_list)
            reasons = flag_manager.get_flag()

            if len(reasons) > 0 or (frame_id + 1) % 9999999999 == 0:

                print("Saving the video")
                output_time = frame_id / fpsmin
                # Convert to real time
                hours = int(output_time // 3600)
                minutes = int(output_time // 60)
                seconds = int(output_time % 60)
                output_time = f"{hours:02d}_{minutes:02d}_{seconds:02d}"
                out_subfolder = os.path.join(args.output_dir, output_time)
                os.makedirs(out_subfolder, exist_ok=True)
                video_paths = [os.path.join(out_subfolder, "{}.mp4".format(idx + 1)) for idx in range(len(frames_list))]

                for index in range(len(frames_list)):
                    out_h, out_w = frames_list[index].shape[:2]
                    out = cv2.VideoWriter(video_paths[index], cv2.VideoWriter_fourcc(*'mp4v'), fpsmin, (out_w, out_h))
                    out_frames = frames_queue_ls[index].get_frames()
                    for f in out_frames:
                        out.write(f)
                    out.release()

                # Save the top view video
                out_h, out_w = top_view_img.shape[:2]
                tv_out = cv2.VideoWriter(os.path.join(out_subfolder, "top_view.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fpsmin, (out_w, out_h))
                out_frames = topview_queue.get_frames()
                for f in out_frames:
                    tv_out.write(f)
                tv_out.release()

                reason_file = os.path.join(out_subfolder, "reason.txt")
                with open(reason_file, 'w') as f:
                    for reason in reasons:
                        f.write(reason + '\n')


            top_row = np.hstack([img_list[1], img_list[0]])
            bottom_row = np.hstack((img_list[3], img_list[2]))
            combined_frame = np.vstack([top_row, bottom_row])
            if args.show_video:
                cv2.imshow("Combined Frame", cv2.resize(combined_frame, (int(real_w*0.8), int(0.8*real_h))))
                cv2.imshow('Top View', top_view_img)
                ch = cv2.waitKey(1)

                color = defaultdict(list)
                text = defaultdict(list)
                # for idx,m in enumerate(merged_dict):
                #     color[idx] = (0,0,255) if merged_dict[m] else (0,255,0)
                #     text[idx] = f"{m}: {merged_dict[m]}"
                #     cv2.putText(combined_frame, text[idx],  (50, 100+50*idx), cv2.FONT_HERSHEY_SIMPLEX, 1, color[idx], 2, cv2.LINE_AA)

            if args.output_video_path:

                vid_writer.write(cv2.resize(combined_frame, (real_w, real_h)))
                topview_writer.write(top_view_img)


            print("Finish processing frame: ", frame_id)
            if frame_id % 100 == 0:
                print(f"Frame {frame_id} processed.")
        else:
            break
        # except:
        #     print("Error processing frame: {}. Skip".format(frame_id))
        #     continue
        frame_id += 1

    print("Video process finished.")
    if args.save_asset:
        if not args.use_saved_box:
            json.dump(box_assets, box_f, indent=4)
        try:
            with open(asset_path, 'w') as f:
                json.dump(assets, f, indent=4)
        except:
            pass


if __name__ == '__main__':
    args = make_parser().parse_args()
    predictor = None if args.use_saved_box else Predictor(args)
    imageflow_demo(predictor, args)


'''
Usage: 
python yolov9_bytetrack_pth_4_cropping.py --video_path /path/to/vidoe --output_dir /path/to/output
'''
