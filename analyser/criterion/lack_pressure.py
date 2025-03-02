from collections import defaultdict
import math
import cv2
from .utils import check_ball_possession,find_closest_player,is_in_rectangle

class LackPressureChecker:

    def __init__(self, max_change_thresh=10,**kwargs):
        self.name = 'lack_pressure'
        self.max_change_thresh = max_change_thresh
        self.flag = False
        self.frame_duration = 10
        self.lack_pressure_list = []
        self.thre = 0.8
        self.lens_h2b = 30# distance
        self.lens_h2h = 30
        self.ball_holder = None
        self.ball_holder_list =[]
        self.catch_list=[]

    def process(self, players,balls,frame_queue, **kwargs):
        court = [(50, 50), (1100, 730)]
        valid_players = defaultdict(list)
        self.flag = False
        if balls:
            ball = balls[0]
            for ball in balls:
                if is_in_rectangle(ball,court):
                    ball = ball
            for p_id, positions in players.items():
                if len(positions) >= self.frame_duration and positions[-1] != [-1,-1] and positions[-self.frame_duration] != [-1,-1]:
                    position = positions[-self.frame_duration:]
                    count = position.count([-1,-1])
                    if count <= self.frame_duration*0.5:
                        valid_players[p_id] = position
            if valid_players:
                min_key_raw, min_distance_raw = find_closest_player(valid_players, ball, -1)
                if min_distance_raw <= self.lens_h2b:
                    self.catch_list.append(min_key_raw)
                else:
                    self.catch_list.append(None)

                if len(self.catch_list) > self.frame_duration:
                    count=0
                    if self.catch_list[-1] != None and \
                            self.catch_list[-self.frame_duration:].count(self.catch_list[-1]) >= self.frame_duration*self.thre:
                        holder = valid_players[min_key_raw][-1]
                        valid2_players = valid_players.pop(min_key_raw)
                        min_key_raw2, min_distance_raw2 = find_closest_player(valid_players, holder, -1)
                        if min_distance_raw2 >= self.lens_h2h:
                            self.lack_pressure_list.append(True)
                            self.ball_holder_list.append(min_key_raw2)
                    else:
                        self.lack_pressure_list.append(False)


                if len(self.lack_pressure_list) >= self.frame_duration and \
                        sum(self.lack_pressure_list[-self.frame_duration:]) >= self.frame_duration*self.thre:
                    self.flag=True
        else:
            self.lack_pressure_list.append(False)


    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'{self.ball_holder_list[-1]} lack of pressure',
                (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'No lack of pressure', (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass




