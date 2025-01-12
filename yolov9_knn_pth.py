import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tennis_court.court_detector import CourtDetector
from tennis_court.top_view import TopViewProcessor
from team_assigner import TeamAssigner

import cv2
import numpy as np

from tracker import BYTETracker

team_colors = []
team_box_colors = [(0, 0, 255), (0, 255, 0)]


def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # analysis = AnalysisManager()

    vid_writer = cv2.VideoWriter(
        args.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    # tracker = BYTETracker(args, frame_rate=30)

    team1_tracker = BYTETracker(args, frame_rate=30)
    team2_tracker = BYTETracker(args, frame_rate=30)

    frame_id = 0
    team_assigner = TeamAssigner()

    # results = []
    mask_points = [(485, 217), (820, 216), (894, 403), (416, 404)]
    top_view = TopViewProcessor(colors=team_box_colors)

    def click_court():
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mask_points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('click_court', frame)
                if len(mask_points) > 4:
                    mask_points.pop(0)
                    print(mask_points)

        height, width, channel = frame.shape
        cv2.namedWindow("click_court", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("click_court", width, height)
        cv2.namedWindow("click_court")
        cv2.setMouseCallback("click_court", click_event)

        while True:
            cv2.imshow("click_court", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        print(mask_points)


    def click_color():
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color = frame[y, x]
                team_colors.append(color)
                # copied_frame = copy.deepcopy(frame)
                # cv2.circle(copied_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('click_color', frame)


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


    while True:
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id == 0:
                click_color()
                team_assigner.assign_color(team_colors)
                click_court()
                court_detector = CourtDetector(mask_points)
                court_detector.begin(type="inner", frame=copy.deepcopy(frame), mask_points=mask_points)
                cv2.destroyAllWindows()

            outputs, img_info = predictor.inference(frame)
            print(outputs.shape)
            # team1_boxes, team2_boxes = [], []
            # team0_boxes, team3_boxes = [], []
            team_boxes = [[] for _ in range(len(team_colors))]

            for output in outputs:
                team_id = team_assigner.get_player_team_test(frame, output[:4],frame_id)
                team_boxes[team_id].append(output)

            colors = team_assigner.team_colors
            for idx, team in enumerate(team_boxes):
                color = colors[idx]
                for box in team:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3]),), color, 2)

            cv2.imshow('Image', frame)
            # cv2.imshow('Top View', cv2.resize(top_view_img, (400, 800)))

            vid_writer.write(frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
        print('frame is',frame_id)


if __name__ == '__main__':
    from yolov9_bytetrack_pth import Predictor, make_parser
    args = make_parser().parse_args()
    predictor = Predictor(args)
    imageflow_demo(predictor, args)
