import cv2
from .utils import calculate_vector, calculate_angle_between_vectors,find_closest_player,compare_motion_direction
import numpy as np

class LackPressureChecker:
    def __init__(self, court, **kwargs):
        self.flag = False
        self.side_referee_coords = []
        self.court = court

    def process(self, players1, players2, balls, **kwargs):  #(542,659), 7.32m=117, 10m=160, 2m=32
        self.team1_dict = players1
        self.team2_dict = players2
        self.balls = balls
        if len(balls):
            ball_position = balls[-1]
            min_key_1, min_distance_1 = find_closest_player(players1, ball_position, -1)
            min_key_2, min_distance_2 = find_closest_player(players2, ball_position, -1)
            distance = np.linalg.norm(np.array(players1[min_key_1][-1]) - np.array(players2[min_key_2][-1]))
            if distance <= 32:
                self.flag = False
            else:
                self.flag = True
    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f"Lack of Pressure", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        elif self.flag == False:
            cv2.putText(frame, f"Normal", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass




