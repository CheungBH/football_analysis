import cv2
from .utils import is_in_rectangle, calculate_ratio
from collections import defaultdict

class GoalKeeperSingleChecker:

    def __init__(self, **kwargs):
        self.name = 'goalkeeper_single'
        self.flag = False
        self.thre = 0.5
        self.frame_duration = 10
        self.door_upper = [0, 0]
        self.door_lower = [0, 0]
        self.attack = 10086
        self.goalkeeper = 10086
        self.attack_team = 10086
        self.flag_list =[]

    def process(self, players,balls, frame_queue, **kwargs):
        self.team_dict = players
        self.flag = False
        court = [(50, 50), (1100, 730)]
        if balls:
            for ball in balls:
                if is_in_rectangle(ball,court):
                    ball = ball
            valid_players = defaultdict(list)
            rect1 = [(50, 320), (210, 586)]
            rect2 = [(960, 320), (1200, 586)]
            rect1_values, rect2_values = [],[]
            rect2_id =[]

            for p_id, positions in players.items():
                if len(positions) >= self.frame_duration and positions[-1] != [-1,-1] and positions[-self.frame_duration] != [-1,-1]:
                    position = positions[-self.frame_duration:]
                    count = position.count([-1,-1])
                    if count <= self.frame_duration*0.5:
                        valid_players[p_id] = position
            if valid_players:
                for v_id,v_positions in valid_players.items():
                    v_position = v_positions[-1]
                    if is_in_rectangle(v_position, rect1):
                        rect1_values.append(v_position)
                    elif is_in_rectangle(v_position, rect2):
                        rect2_values.append(v_position)
                        rect2_id.append(v_id)

            if len(rect1_values) == 2 and is_in_rectangle(ball,rect1):
                ratio = calculate_ratio(rect1_values[0][0], rect1_values[1][0], 50)
                if ratio < 0.5:
                    self.flag_list.append(True)
                else:
                    self.flag_list.append(False)
            elif len(rect2_values) == 2 and is_in_rectangle(ball,rect2):
                ratio = calculate_ratio(rect2_values[0][0], rect2_values[1][0], 1100)
                if ratio < 0.5:
                    self.flag_list.append(True)
                else:
                    self.flag_list.append(False)
            else:
                self.flag_list.append(False)

            if len(self.flag_list) >= self.frame_duration:
                if sum(self.flag_list[-self.frame_duration:]) >= self.frame_duration*self.thre:
                    self.flag = True
            else:
                self.flag = False


    def visualize(self, frame, idx):
        if self.flag== True:
            cv2.putText(frame, "GKem Scores low", (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "GKem Scores high", (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)


    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
        pass

