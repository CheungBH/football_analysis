import cv2
from collections import defaultdict
from .utils import calculate_vector, calculate_angle_between_vectors,find_closest_player,compare_motion_direction,vector_angle
import math

class MovingReverseChecker:
    def __init__(self, **kwargs):
        self.angle_threshold = 135
        self.frame_duration = 10
        self.thre=0.6
        self.flag = False
        self.colors = [(0, 0, 255), (125, 125, 125)]
        self.curve_duration = 10
        self.base_vector = calculate_vector((0, 0), (0, 1))
        self.flaglist=[]

    # def process(self, players1, players2, balls, **kwargs):
    #     self.team1_dict = players1
    #     self.team2_dict = players2
    #     self.balls = balls
    #     if len(balls):
    #         ball_position = balls[-1]
    #         min_key_1, min_distance_1 = find_closest_player(players1, ball_position, -1)
    #         min_key_2, min_distance_2 = find_closest_player(players2, ball_position, -1)
    #         if len(self.team1_dict[min_key_1])>self.frame_duration and len(self.team2_dict[min_key_2])>self.frame_duration:
    #             if min_distance_1 < min_distance_2: #red offense,gray defense
    #                 Attack_list = self.team1_dict[min_key_1][-self.frame_duration:]
    #                 Defend_list = self.team2_dict[min_key_2][-self.frame_duration:]
    #                 attack=1
    #             else:
    #                 Attack_list = self.team2_dict[min_key_2][-self.frame_duration:]
    #                 Defend_list = self.team1_dict[min_key_1][-self.frame_duration:]
    #                 attack=2
    #             if Attack_list[-1][1]<Defend_list[-1][1]:
    #                 Door_list=[83,542]
    #             else:
    #                 Door_list=[83,662]
    #             self.flag = compare_motion_direction(Attack_list,Defend_list,Door_list)
    #             self.flaglist.append(self.flag)

        ''' 
        for team_id, team in enumerate([players1, players2]):
            for p_id, position in team.items():
                if len(position) >= self.frame_duration:
                    vector = calculate_vector(position[0], position[-1])
                    angle = calculate_angle_between_vectors(self.base_vector, vector)
                    #
                    self.angles[team_id][p_id] = angle
                    if angle > self.angle_threshold:
                        self.flag = True
        '''
    def process(self, players,frame_queue, **kwargs):
        self.team_dict = players
        self.reverse_list=[]
        self.flag = False
        key_vectors = {}
        for key, values in self.team_dict.items():
            if len(values) >= frame_queue:
                first_vector = values[-frame_queue]
                last_vector = values[-1]
                key_vectors[key] = (first_vector, last_vector)

        for key1, (first_vec1, last_vec1) in key_vectors.items():
            for key2, (first_vec2, last_vec2) in key_vectors.items():
                if key1 != key2:
                    angle = vector_angle(last_vec1, first_vec2)
                    if angle > 120:
                        self.reverse_list.append([key1,key2])
                        self.flag = True


    def visualize(self, frame):
        if self.flag== True:
            cv2.putText(frame, "Someone reverse", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)
            for idx,reverse in enumerate(self.reverse_list):
                cv2.putText(frame, "ID {} and ID{} is reverse".format(self.reverse_list[idx][0],self.reverse_list[idx][1]),
                            (500, 100 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No reverse", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass
        '''
        for t_idx, (color, team_dict) in enumerate(zip(self.colors, [self.team1_dict, self.team2_dict])):
            for p_idx, (player, locations) in enumerate(team_dict.items()):
                angle = self.angles[t_idx][player]
                cv2.putText(frame, "id {}: Angle {}".format(player, angle), (100 + t_idx * 500, 100 + p_idx * 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
        '''

    def vis_path(self, frame, locations, vis_duration, color):
        for i in range(vis_duration):
            cv2.circle(frame, (int(locations[-i][0]), int(locations[-i][1])), 20, color, -1)
        for j in range(vis_duration-1):
            cv2.line(frame, (int(locations[-j][0]), int(locations[-j][1])),
                     (int(locations[-(j-1)][0]), int(locations[-j-1][1])), color, 3)
