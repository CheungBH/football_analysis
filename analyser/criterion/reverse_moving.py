import cv2
from collections import defaultdict
from .utils import calculate_vector, calculate_angle_between_vectors,find_closest_player,compare_motion_direction,vector_angle,check_speed_distance,is_in_rectangle,calculate_speed,is_within_radius


class MovingReverseChecker:
    def __init__(self, **kwargs):
        self.name = 'reverse_moving'
        self.angle_threshold = 135
        self.frame_duration = 10
        self.thre=0.6
        self.flag = False
        self.colors = [(0, 0, 255), (125, 125, 125)]
        self.curve_duration = 10
        self.base_vector = calculate_vector((0, 0), (0, 1))
        self.flaglist=[]
        self.reverse_count = defaultdict(list)


    def process(self, players,balls,frame_queue, **kwargs):
        self.team_dict = players
        self.reverse_list = []
        self.flag = False
        self.frame_duration = frame_queue
        human_valid = defaultdict(list)
        court = [(50, 50), (1100, 730)]
        key_vectors = {}

        if balls:
            ball = balls[0]
            for ball in balls:
                if is_in_rectangle(ball,court):
                    ball = ball

            for p_id, position in players.items():
                if len(position) >= self.frame_duration and position[-1] != [-1,-1] and position[-self.frame_duration]!= [-1,-1]:
                    if is_within_radius(position[-1], ball, 100): #10m
                        speeds = calculate_speed(position[-self.frame_duration:])
                        low_speed_count = sum(1 for speed in speeds if speed < 1)
                        if low_speed_count <=10:
                            vector_valid = [position[-1][0]-position[-self.frame_duration][0],
                                            position[-1][1]-position[-self.frame_duration][1]]
                            human_valid[p_id] = vector_valid
            if len(human_valid) >=2:
                for h1,v1 in human_valid.items():
                    for h2,v2 in human_valid.items():
                        if h1<h2:
                            angle = vector_angle(v1, v2)
                            if angle > 120:
                                if (h1, h2) not in self.reverse_count:
                                    self.reverse_count[(h1, h2)] = 0
                                self.reverse_count[(h1, h2)] += 1

                                if self.reverse_count[(h1, h2)] >= self.frame_duration:
                                    self.reverse_list.append([h1, h2])
                                    self.flag = True
                                    #del(self.reverse_count[(key1, key2)])
                            else:
                                if (h1, h2) in self.reverse_count:
                                    self.reverse_count[(h1, h2)] = 0

            # for key1, key_vector1 in key_vectors.items():
            #     for key2, key_vector2 in key_vectors.items():
            #         if key1 < key2:
            #             angle = vector_angle(key_vector1, key_vector2)
            #             if angle > 120:
            #                 if (key1, key2) not in self.reverse_count:
            #                     self.reverse_count[(key1, key2)] = 0
            #                 self.reverse_count[(key1, key2)] += 1
            #
            #                 if self.reverse_count[(key1, key2)] >= self.frame_duration:
            #                     self.reverse_list.append([key1, key2])
            #                     self.flag = True
            #                     #del(self.reverse_count[(key1, key2)])
            #             else:
            #                 if (key1, key2) in self.reverse_count:
            #                     self.reverse_count[(key1, key2)] = 0


    def visualize(self, frame):
        if self.flag== True:
            cv2.putText(frame, "Someone reverse", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
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
