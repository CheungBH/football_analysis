import cv2
from collections import defaultdict
from .utils import calculate_vector, calculate_angle_between_vectors, find_closest_player, polygon_area
from .utils import is_in_rectangle,calculate_ratio
import numpy as np


class GoalKeeperSingleChecker:
    def __init__(self, **kwargs):
        self.flag = False
        self.frame_duration = 10
        self.door_upper = [0, 0]
        self.door_lower = [0, 0]
        self.attack = 10086
        self.goalkeeper = 10086
        self.attack_team = 10086

    # def process(self, players1, players2, goalkeepers, balls, matrix, **kwargs):
    #     self.team1_dict = players1
    #     self.team2_dict = players2
    #     self.goalkeepers_dict = goalkeepers
    #     self.balls = balls
    #     self.inverse_matrix = np.linalg.inv(matrix)
    #     self.door_upper = cv2.perspectiveTransform(np.array([[[83.0, 542.0]]]), self.inverse_matrix)[0][0].tolist()
    #     self.door_lower = cv2.perspectiveTransform(np.array([[[83.0, 662.0]]]), self.inverse_matrix)[0][0].tolist()
    #     if balls:
    #         ball_position = balls[-1]
    #         min_key_1, min_distance_1 = find_closest_player(players1, ball_position, -1)
    #         min_key_2, min_distance_2 = find_closest_player(players2, ball_position, -1)
    #         self.goalkeeper, _ = find_closest_player(goalkeepers, ball_position, -1)
    #         if len(players1[min_key_1]) > self.frame_duration and len(players2[min_key_2]) > self.frame_duration:
    #             self.attack, self.attack_team = (min_key_1, 0) if min_distance_1 < min_distance_2 else (min_key_2, 1)
    #         attack_point = self.team1_dict[self.attack][-1] if self.attack_team == 0 else self.team2_dict[self.attack][-1]
    #         defend_point = self.goalkeepers_dict[self.goalkeeper][-1]
    #         self.attack_rgb_point = cv2.perspectiveTransform(np.array([[attack_point]]), self.inverse_matrix)[0][0].tolist()
    #         self.defend_rgb_point = cv2.perspectiveTransform(np.array([[defend_point]]), self.inverse_matrix)[0][0].tolist()
    #         self.area1 = np.array([self.attack_rgb_point, self.door_upper, self.door_lower], np.int32).reshape((-1, 1, 2))
    #         self.area2 = np.array([self.defend_rgb_point, self.door_upper, self.door_lower], np.int32).reshape((-1, 1, 2))
    #         self.ratio = polygon_area(self.area2)/polygon_area(self.area1)

    def process(self, players,balls, matrix, **kwargs):
        self.team_dict = players
        rect1 = [(50, 184), (215, 586)]
        rect2 = [(935, 184), (1100, 586)]
        rect1_values, rect2_values = [],[]
        for key, values in self.team_dict.items():
            value = values[-1]
            if is_in_rectangle(value, rect1):
                rect1_values.append(value)
            elif is_in_rectangle(value, rect2):
                rect2_values.append(value)

        # 检查每个长方形区域中是否只有两个value，并计算比值
        if len(rect1_values) == 2:
            ratio = calculate_ratio(rect1_values[0][0], rect1_values[1][0], 50)
            if ratio >0.5:
                self.flag = False
            else:
                self.flag = True
        elif len(rect2_values) == 2:
            ratio = calculate_ratio(rect2_values[0][0], rect2_values[1][0], 1100)
            if ratio >0.5:
                self.flag = False
            else:
                self.flag = True


    def visualize(self, frame):
        if self.flag== True:
            cv2.putText(frame, "GKem Scores low", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Good Defense", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)
        # if self.attack_team != 10086:
        #     overlay = frame.copy()
        #     cv2.fillPoly(overlay, [self.area1], (0, 0, 255))
        #     cv2.fillPoly(overlay, [self.area2], (255, 0, 0))
        #     cv2.addWeighted(overlay, 0.1, frame, 0.9, 0.1, frame)
        #     if self.ratio > 0.5:
        #         cv2.putText(frame, 'Good Defense', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                                          (0, 255, 0), 2, cv2.LINE_AA)
        #     cv2.putText(frame, f'Ratio: {self.ratio:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
        #                     cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
