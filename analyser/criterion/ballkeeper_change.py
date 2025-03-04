from collections import defaultdict
import math
import cv2
from .utils import check_ball_possession,find_closest_player,is_in_rectangle
class BallKeeperChangeChecker:

    def __init__(self, max_change_thresh=10,**kwargs):
        self.name = 'ballkeeper_change'
        self.max_change_thresh = max_change_thresh
        self.flag = False
        self.frame_duration = 10
        self.catch_list = []
        self.last_holder= None
        self.thre = 0.8
        self.lens_h2h = 30# distance
        self.lens_h2b = 40
        self.ball_holder = None
        self.ball_holder_list =[]
        self.ball_change_time = 0

    '''
    def process(self, players, balls, frame_queue, **kwargs):
    '''

    def process(self, players,balls, frame_queue, **kwargs):
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
                catch_position = valid_players[min_key_raw][-1]
                if min_distance_raw <= self.lens_h2b:
                    valid_else = valid_players.pop(min_key_raw)
                    min_key_raw2, min_distance_raw2 = find_closest_player(valid_players, catch_position, -1)
                    if min_distance_raw2 >= self.lens_h2h:
                        self.catch_list.append(min_key_raw)
                    else:
                        self.catch_list.append(None)
                else:
                    self.catch_list.append(None)

                if len(self.catch_list) > self.frame_duration:
                    count=0
                    if self.catch_list[-1] != None and \
                            self.catch_list[-self.frame_duration:].count(self.catch_list[-1]) >= self.frame_duration*self.thre:
                        self.ball_holder_list.append(self.catch_list[-1])

                if len(self.ball_holder_list) >=2 and self.catch_list[-1] is not None:
                    if self.ball_holder_list[-1] != self.ball_holder_list[-2]:
                        self.flag = True
                        self.ball_change_time += 1
                        #print(str(self.ball_holder_list[-2]) +"pass the ball to" + str(self.last_holder[-1]))
                #print(self.catch_list)
                print(self.ball_holder_list)
        else:
            self.catch_list.append(None)
            # if self.last_holder == None and self.ball_holder != None:
            #     print(str(self.ball_holder) + " hold the ball")
            #     self.last_holder = self.ball_holder
            # elif self.last_holder != None and self.ball_holder != self.last_holder:
            #     self.flag = True
            #     print(str(self.last_holder) +"pass the ball to" + str(self.last_holder))
            #     self.last_holder = self.ball_holder

    def visualize(self,frame, idx):

        if self.flag == True:
            cv2.putText(frame, f'Ball change from {self.ball_holder_list[-2]} to {self.ball_holder_list[-1]}',
                (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'Ball no change', (100, 100+(idx*40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Ball change time: {}".format(self.ball_change_time), (100, 140+(idx*40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
        pass

