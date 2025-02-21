import cv2
from collections import defaultdict
from .utils import check_speed_displacement


class LowSpeedChecker:
    def __init__(self, **kwargs):
        self.speed_threshold_low = 0.7
        self.speed_threshold_nomove = 0.4
        self.frame_duration = 10
        self.flag = 1
        self.colors = [(0, 0, 255), (125, 125, 125)]
        self.curve_duration = 10

    def process(self, players1, players2, balls,frame_queue, **kwargs):
        self.team1_dict = players1
        self.team2_dict = players2
        self.frame_duration = frame_queue
        self.flag = 1
        self.balls = balls
        self.speeds = [defaultdict(float), defaultdict(float)]
        self.low_speed_players = [[], []]
        self.nomove_players = [[], []]

        for team_id, team in enumerate([players1, players2]):
            for p_id, position in team.items():
                if len(position) >= self.frame_duration: # *ratio
                    speed = check_speed_displacement(position[-self.frame_duration:])
                    self.speeds[team_id][p_id] = speed
                    if speed < self.speed_threshold_low:
                        self.flag = 0
                        self.low_speed_players[team_id].append(p_id)
                    else:
                        self.flag = 1

    def visualize(self, frame):
        if self.flag == 0:
            cv2.putText(frame, "Normal speed", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif self.flag == 1:
            cv2.putText(frame, "Low Speed", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif self.flag == 2:
            cv2.putText(frame, "No moving", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        for team_id, team_players in enumerate(self.low_speed_players):
            for p_id in team_players:
                    cv2.putText(frame, "ID {} is low speed".format(p_id), (100, 100 + p_id * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        for team_id, team_players in enumerate(self.nomove_players):
            for p_id in team_players:
                cv2.putText(frame, "ID {} is no moving".format(p_id), (400, 100 + p_id * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

    def vis_path(self, frame, locations, vis_duration, color):
        for i in range(vis_duration):
            cv2.circle(frame, (int(locations[-i][0]), int(locations[-i][1])), 20, color, -1)
        for j in range(vis_duration - 1):
            cv2.line(frame, (int(locations[-j][0]), int(locations[-j][1])),
                     (int(locations[-(j + 1)][0]), int(locations[-(j + 1)][1])), color, 3)
