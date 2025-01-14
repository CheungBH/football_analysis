import cv2
from collections import defaultdict
from .utils import calculate_vector, calculate_angle_between_vectors


class MovingReverseChecker:
    def __init__(self, **kwargs):
        self.angle_threshold = 135
        self.frame_duration = 10
        self.flag = False
        self.colors = [(0, 0, 255), (125, 125, 125)]
        self.curve_duration = 10
        self.base_vector = calculate_vector((0, 0), (0, 1))

    def process(self, players1, players2, balls, **kwargs):
        self.team1_dict = players1
        self.team2_dict = players2
        self.balls = balls
        self.angles = [defaultdict(float), defaultdict(float)]
        for team_id, team in enumerate([players1, players2]):
            for p_id, position in team.items():
                if len(position) >= self.frame_duration:
                    vector = calculate_vector(position[0], position[-1])
                    angle = calculate_angle_between_vectors(self.base_vector, vector)
                    self.angles[team_id][p_id] = angle
                    if angle > self.angle_threshold:
                        self.flag = True

    def visualize(self, frame):
        if not self.flag:
            cv2.putText(frame, "Normal", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Someone is reverse", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)

        for t_idx, (color, team_dict) in enumerate(zip(self.colors, [self.team1_dict, self.team2_dict])):
            for p_idx, (player, locations) in enumerate(team_dict.items()):
                angle = self.angles[t_idx][player]
                cv2.putText(frame, "id {}: Angle {}".format(player, angle), (100 + t_idx * 500, 100 + p_idx * 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)

    def vis_path(self, frame, locations, vis_duration, color):
        for i in range(vis_duration):
            cv2.circle(frame, (int(locations[-i][0]), int(locations[-i][1])), 20, color, -1)
        for j in range(vis_duration-1):
            cv2.line(frame, (int(locations[-j][0]), int(locations[-j][1])),
                     (int(locations[-(j-1)][0]), int(locations[-j-1][1])), color, 3)
