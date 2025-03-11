from collections import defaultdict
import math
from collections import Counter
import cv2
from .utils import check_ball_possession,find_closest_player,is_in_rectangle,calculate_speed

class OutpaceChecker:

    def __init__(self, court, display_x=100, **kwargs):
        self.name = 'outpace'
        self.display_x = display_x
        self.field = court
        self.ball_coords = []
        self.flag = False
        self.flag_list=[]
        self.thre = 0.7
        self.frame_duration = 10
        self.catch_list = []
        self.last_holder= None
        self.lens_h2h = 10# distance
        self.lens_h2b = 45
        self.ball_holder = None
        self.ball_holder_list =[]
        self.ball_holder_color_list = []
        self.outpace_thre = 4.0

    def process(self, players,balls, color, **kwargs):
        court = [(50, 50), (1100, 730)]
        invalid_area =[(500, 60), (750, 230)]
        valid_players = defaultdict(list)
        valid_colors = defaultdict(list)
        self.flag = False
        if balls:
            ball = balls[0]
            for ball in balls:
                if is_in_rectangle(ball,court):
                    ball = ball
            for p_id, positions in players.items():
                if len(positions) >= self.frame_duration and positions[-1] != [-1,-1] \
                        and not is_in_rectangle(positions[-1],invalid_area) and positions[-self.frame_duration] != [-1,-1]:
                    position = positions[-self.frame_duration:]
                    count = position.count([-1,-1])
                    if count <= self.frame_duration*0.5:
                        valid_players[p_id] = position
                        valid_colors[p_id] = color
            if valid_players:
                min_key_raw, min_distance_raw = find_closest_player(valid_players, ball, -1)
                catch_position = valid_players[min_key_raw][-1]
                if min_distance_raw <= self.lens_h2b:
                    valid_else = valid_players.pop(min_key_raw)
                    if valid_players:
                        min_key_raw2, min_distance_raw2 = find_closest_player(valid_players, catch_position, -1)
                        if min_distance_raw2 >= self.lens_h2h:
                            self.catch_list.append(min_key_raw)
                        else:
                            self.catch_list.append(None)
                    else:
                        self.catch_list.append(min_key_raw)
                else:
                    self.catch_list.append(None)

                if len(self.catch_list) > self.frame_duration:
                    if self.catch_list[-1] is not None:
                        if not self.ball_holder_list or (self.ball_holder_list[-1] != self.catch_list[-1]):
                            if self.catch_list[-self.frame_duration:].count(self.catch_list[-1]) >= self.frame_duration * self.thre:
                                self.ball_holder_list.append(self.catch_list[-1])
                                filter_color_list = [x for x in color[min_key_raw] if x != -1]
                                counter = Counter(filter_color_list)
                                team_color = counter.most_common(1)[0][0] if counter else 0
                                self.ball_holder_color_list.append(team_color)
            if self.ball_holder_list:
                rest_speeds = []
                holder_position = players[self.ball_holder_list[-1]][-self.frame_duration:]
                holder_speed = calculate_speed(holder_position)
                if holder_speed > self.outpace_thre:
                    for human,rest_position in valid_players.items():
                        rest_speeds.append(calculate_speed(rest_position))
                    self.flag_list.append(any(rest_speed < holder_speed for rest_speed in rest_speeds))
                    if sum(self.flag_list[-self.frame_duration:]) > 8:
                        self.flag=True
                else:
                    self.flag_list.append(False)
            else:
                self.flag_list.append(False)
        else:
            self.flag_list.append(False)



    def visualize(self, frame, idx):
        if self.flag == True:
            cv2.putText(frame, f'Outpace', (self.display_x, 100+(idx*40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'Normal Pace', (self.display_x, 100+(idx*40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
        pass


if __name__ == '__main__':
    import time
