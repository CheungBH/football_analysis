import cv2
from collections import defaultdict,Counter
from .utils import check_speed_displacement ,calculate_speed,is_in_rectangle, is_within_radius


class SpeedChecker:
    def __init__(self, **kwargs):
        self.name = 'low_speed_with_ball'
        self.speed_threshold = 1.45
        # self.speed_threshold_nomove = 0.3
        self.frame_duration = 50
        self.thre = 0.8
        self.flag_low = 0
        self.average_speed = []
        self.flag = False
        self.curve_duration = 10
        self.flag_list=[[], []]
        self.low_speed_players = []
        self.nomove_players = []
        self.team_dict = {}
        self.green_word = "Normal speed"
        self.red_word = "Low speed"
        self.detail_word = "low speed"


    def process(self, players, balls,frame_queue,color, **kwargs):
        court = [(50, 50), (1100, 730)]
        invalid_area =[(500, 80), (755, 230)]

        if balls:
            for ball in balls:
                if is_in_rectangle(ball, court):
                    ball = ball
        else:
            self.flag = False
            return

        self.flag = False
        self.flag_low = 0
        valid_players = defaultdict(list)
        self.speeds = [defaultdict(float),defaultdict(float)]
        self.target_players = []
        average_speed_single = []
        # self.nomove_players = []

        for p_id, positions in players.items():
            if len(positions) >= self.frame_duration and positions[-1] != [-1,-1] and positions[-self.frame_duration] != [-1,-1]:
                if not is_within_radius(positions[-1], ball, 100) or is_in_rectangle(ball,invalid_area):
                    continue
                position = positions[-self.frame_duration:]
                count = position.count([-1,-1])
                if count <= self.frame_duration*0.5:
                    speeds = calculate_speed(position)
                    average_speed_single.append(sum(speeds)/len(speeds))
                    valid_players[p_id] = position
                    filter_color_list = [x for x in color[p_id] if x != -1]
                    counter = Counter(filter_color_list)
                    team_color = counter.most_common(1)[0][0] if counter else 0
                    #speeds = [check_speed_displacement(position[i:i+10]) for i in range(len(position) - 10)]

                    low_speed_count = sum(1 for speed in speeds if speed < self.speed_threshold)
                    if low_speed_count >= len(speeds) * self.thre and team_color in (0,1):
                        self.target_players.append(p_id)
                        self.flag_low += 1
        if average_speed_single:
            self.average_speed.append(sum(average_speed_single)/len(average_speed_single))
        if self.flag_low > 1:
            self.flag = True

    def visualize(self, frame, idx):
        if self.flag == False:
            cv2.putText(frame, self.green_word, (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, self.red_word, (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # elif self.flag == 2:
        #     cv2.putText(frame, "No moving", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
        for idx,p_id in enumerate(self.low_speed_players):
            cv2.putText(frame, "ID {} is {}".format(p_id, self.detail_word), (300, 100 + idx * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)



    def vis_path(self, frame, locations, vis_duration, color):
        for i in range(vis_duration):
            cv2.circle(frame, (int(locations[-i][0]), int(locations[-i][1])), 20, color, -1)
        for j in range(vis_duration - 1):
            cv2.line(frame, (int(locations[-j][0]), int(locations[-j][1])),
                     (int(locations[-(j + 1)][0]), int(locations[-(j + 1)][1])), color, 3)
