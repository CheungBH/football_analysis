import cv2
from .utils import check_speed_ls
from collections import defaultdict


class StandingChecker:
    def __init__(self, **kwargs):
        self.speed_threshold = 0.1
        self.frame_duration = 10
        self.flag = False
        self.colors = [(0, 0, 255), (125, 125, 125)]
        self.curve_duration = 10
        # self.team1

    def process(self, players1, players2, balls, **kwargs):
        self.team1_dict = players1
        self.team2_dict = players2
        self.balls = balls
        self.flag = False
        self.speeds = [defaultdict(float), defaultdict(float)]
        for team_id, team in enumerate([players1, players2]):
            for p_id, position in team.items():
                if len(position) >= self.frame_duration:
                    speed = check_speed_ls(position[-self.frame_duration:])
                    self.speeds[team_id][p_id] = speed
                    if speed < self.speed_threshold:
                        self.flag = True

    def visualize(self, frame):
        if not self.flag:
            cv2.putText(frame, "Normal speed", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Standing no move", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)

        for t_idx, (color, team_dict) in enumerate(zip(self.colors, [self.team1_dict, self.team2_dict])):
            for p_idx, (player, locations) in enumerate(team_dict.items()):
                # vis_duration = min(len(locations), self.curve_duration)
                # self.vis_path(frame, locations, vis_duration, color)
                speed = round(self.speeds[t_idx][player], 5)
                cv2.putText(frame, "id {}: Speed {}".format(player, speed), (100 + t_idx * 500, 100 + p_idx * 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)

    def vis_path(self, frame, locations, vis_duration, color):
        for i in range(vis_duration):
            cv2.circle(frame, (int(locations[-i][0]), int(locations[-i][1])), 20, color, -1)
        for j in range(vis_duration-1):
            cv2.line(frame, (int(locations[-j][0]), int(locations[-j][1])),
                     (int(locations[-(j-1)][0]), int(locations[-j-1][1])), color, 3)
