from collections import defaultdict
import math
import cv2
from .utils import check_ball_possession,find_closest_player
class BallKeeperChangeChecker:
    def __init__(self, max_change_thresh=10,**kwargs):
        self.max_change_thresh = max_change_thresh
        self.flag = False
        self.catcher = 1
        self.first_catcher = 1
        self.last_holder = 1
        self.catch_list = []
        self.last_catch_list = []
        self.thre = 5
        self.holder = 0

    def process(self, players1, players2, balls, **kwargs):
        if len(balls):
            ball_position = balls[-1]
            # 找到第一次接球者(上一次接球)
            min_key_raw1, min_distance_raw1 = find_closest_player(players1, ball_position, 0)
            min_key_raw2, min_distance_raw2 = find_closest_player(players2, ball_position, 0)
            self.first_catcher = 1 if min_distance_raw1 < min_distance_raw2 else 2

            # 找到最终接球者
            min_key_1, min_distance_1 = find_closest_player(players1, ball_position, -1)
            min_key_2, min_distance_2 = find_closest_player(players2, ball_position, -1)
            self.catcher = 1 if min_distance_1 < min_distance_2 else 2
            self.catch_list.append(self.catcher)
            if len(self.catch_list) > 5:
                self.holder, self.flag = check_ball_possession(self.catch_list,self.thre)


    def visualize(self, frame):
        if self.holder == 1:
            cv2.putText(frame, f'Team1 catch the ball', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        elif self.holder == 2:
            cv2.putText(frame, f'Team2 catch the ball', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        elif self.holder == 0:
            cv2.putText(frame, f'Fighting for the ball', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        if self.flag == True:
            cv2.putText(frame, f'Ball change', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass

