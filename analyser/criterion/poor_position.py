import cv2
from .utils import is_in_rectangle, calculate_ratio,is_point_in_triangle
from collections import defaultdict,Counter

class PoorPositionChecker:
    def __init__(self, **kwargs):
        self.name = 'poor_position'
        self.flag = False
        self.thre = 0.5
        self.frame_duration = 10
        self.flag_list =[]

    def process(self, players,balls, color,frame_queue, **kwargs):
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
            rect1_ball = [(50, 50), (210, 720)]
            rect2_ball = [(960,50), (1200, 720)]
            rect1_values, rect2_values = [],[]
            rect1_id,rect2_id =[],[]
            rect1_color,rect2_color=[],[]

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
                        rect1_id.append(v_id)
                        filter_color_list = [x for x in color[v_id] if x != -1]
                        counter = Counter(filter_color_list)
                        team_color = counter.most_common(1)[0][0] if counter else 0
                        rect1_color.append(team_color)
                    elif is_in_rectangle(v_position, rect2):
                        rect2_values.append(v_position)
                        rect2_id.append(v_id)
                        filter_color_list = [x for x in color[v_id] if x != -1]
                        counter = Counter(filter_color_list)
                        team_color = counter.most_common(1)[0][0] if counter else 0
                        rect2_color.append(team_color)
            referee = 4
            if 1<len(rect1_values) < 4 and is_in_rectangle(ball,rect1_ball) and referee not in rect1_color: # ball_position
                goal_keeper = min(rect1_values, key=lambda point: point[0])
                upper_door = [50,349]
                lower_door = [50,421]
                self.flag_list.append(not is_point_in_triangle(goal_keeper,ball,upper_door,lower_door))
            elif 1<len(rect2_values) < 4 and is_in_rectangle(ball,rect2_ball) and referee not in rect2_color:
                goal_keeper = max(rect2_values, key=lambda point: point[0])
                upper_door = [1100,349]
                lower_door = [1100,421]
                self.flag_list.append(not is_point_in_triangle(goal_keeper,ball,upper_door,lower_door))
            else:
                self.flag_list.append(False)

            if len(self.flag_list) >= self.frame_duration:
                if sum(self.flag_list[-self.frame_duration:]) >= self.frame_duration*self.thre:
                    self.flag = True
            else:
                self.flag = False
        else:
            self.flag_list.append(False)


    def visualize(self, frame, idx):
        if self.flag== True:
            cv2.putText(frame, "Poor Position", (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Normal Position", (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)


    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
        pass

