from collections import defaultdict
import math
import cv2
from .utils import check_ball_possession,find_closest_player
class BallKeeperChangeChecker:

    def __init__(self, max_change_thresh=10,**kwargs):
        self.name = 'ballkeeper_change'
        self.max_change_thresh = max_change_thresh
        self.flag = False
        self.frame_duration = 5
        self.catch_list = []
        self.last_holder= None
        self.thre = 0.8
        self.lens = 100# distance
        self.ball_holder = None
        self.ball_holder_list =[]

    def process(self, players,balls,frame_queue, **kwargs):
        if len(balls):
            self.flag = False
            ball_position = balls[-1]
            min_key_raw, min_distance_raw = find_closest_player(players, ball_position, 0)
            if min_distance_raw <= self.lens:
                self.catch_list.append(min_key_raw)
            else:
                self.catch_list.append(None)

            if len(self.catch_list) > self.frame_duration:
                count=0
                for catcher in self.catch_list[-self.frame_duration:]:
                    if catcher == self.catch_list[-1] and catcher!= None:
                        count += 1
                if count >= self.frame_duration* self.thre:
                    if self.ball_holder_list ==[]:
                        self.ball_holder_list.append(min_key_raw)
                    else:
                        if min_key_raw != self.ball_holder_list[-1]:
                            self.ball_holder_list.append(min_key_raw)
            if len(self.ball_holder_list) >=2:
                if self.ball_holder_list[-1] != self.ball_holder_list[-2]: #and self.ball_holder_list[-2]!=self.ball_holder_list[-3] and self.ball_holder_list[-1] != self.ball_holder_list[-3]:
                    self.flag = True
                    #print(str(self.ball_holder_list[-2]) +"pass the ball to" + str(self.last_holder[-1]))
            print(self.ball_holder_list)
            # if self.last_holder == None and self.ball_holder != None:
            #     print(str(self.ball_holder) + " hold the ball")
            #     self.last_holder = self.ball_holder
            # elif self.last_holder != None and self.ball_holder != self.last_holder:
            #     self.flag = True
            #     print(str(self.last_holder) +"pass the ball to" + str(self.last_holder))
            #     self.last_holder = self.ball_holder

    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'Ball change from {self.ball_holder_list[-2]} to {self.ball_holder_list[-1]}',
                (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'Ball no change', (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass

