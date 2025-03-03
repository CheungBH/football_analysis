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
        self.ball_list=[]


    def process(self, players,balls,frame_queue, **kwargs):
        self.team_dict = players
        valid_players = defaultdict(list)
        self.reverse_list = []
        self.flag = False
        #self.frame_duration = frame_queue
        human_valid = defaultdict(list)
        court = [(50, 50), (1100, 730)]
        key_vectors = {}

        if balls:
            ball = balls[0]
            for ball in balls:
                if is_in_rectangle(ball,court):
                    ball = ball
                    self.ball_list.append(ball)
                    #ball_count = sum(1 for ball in self.ball_list[-self.frame_duration:] if ball != [-1,-1])


            for p_id, positions in players.items():
                if len(positions) >= self.frame_duration and positions[-1] != [-1,-1] and positions[-self.frame_duration]!= [-1,-1]:
                    position = positions[-self.frame_duration:]
                    count = position.count([-1,-1])
                    if count <= self.frame_duration*0.5:
                        valid_players[p_id] = position
            if valid_players:
                for v_id,v_positions in valid_players.items():
                    if is_within_radius(v_positions[-1], ball, 150) and not is_within_radius(v_positions[-1], ball, 50): #220,250 for 24
                        speeds = calculate_speed(v_positions[-self.frame_duration:])
                        low_speed_count = sum(1 for speed in speeds if speed < 2) # spped thre
                        if low_speed_count <= self.frame_duration*self.thre:
                            vector_valid = [v_positions[-1][0]-v_positions[-self.frame_duration][0],
                                            v_positions[-1][1]-v_positions[-self.frame_duration][1]]
                            human_valid[v_id] = vector_valid
            if len(self.ball_list) > self.frame_duration and human_valid:
                ball_vec = [self.ball_list[-1][0] - self.ball_list[-self.frame_duration][0],
                            self.ball_list[-1][1] - self.ball_list[-self.frame_duration][1]]
                #ball_vecs = [[self.ball_list[i+1][0] - self.ball_list[i][0], self.ball_list[i+1][1] - self.ball_list[i][1]] for i in range(len(self.ball_list) - 1)]
                for h,huamen_vec in human_valid.items():
                    angle = vector_angle(huamen_vec, ball_vec)
                    if angle > 120:
                        if h not in self.reverse_count:
                            self.reverse_count[h] = 0
                        self.reverse_count[h] += 1
                        if self.reverse_count[h] >= self.frame_duration*self.thre:
                                self.reverse_list.append(h)
                                self.flag = True
                    else:
                        if h in self.reverse_count:
                            self.reverse_count[h] = 0
        else:
            ball_last =self.ball_list[-1] if self.ball_list else [-1,-1]
            self.ball_list.append(ball_last)
            if len(self.ball_list)>= self.frame_duration:
                if self.ball_list[-self.frame_duration] == self.ball_list[-1]:
                    self.ball_list=[]

            # if len(human_valid) >=2: # human ball vector
            #     for h1,v1 in human_valid.items():
            #         for h2,v2 in human_valid.items():
            #             if h1<h2:
            #                 angle = vector_angle(v1, v2)
            #                 if angle > 120:
            #                     if (h1, h2) not in self.reverse_count:
            #                         self.reverse_count[(h1, h2)] = 0
            #                     self.reverse_count[(h1, h2)] += 1
            #
            #                     if self.reverse_count[(h1, h2)] >= 1:#self.frame_duration*self.thre:
            #                         self.reverse_list.append([h1, h2])
            #                         self.flag = True
            #                         #del(self.reverse_count[(key1, key2)])
            #                 else:
            #                     if (h1, h2) in self.reverse_count:
            #                         self.reverse_count[(h1, h2)] = 0



    def visualize(self, frame, idx):
        if self.flag== True:
            cv2.putText(frame, "Someone reverse", (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2, cv2.LINE_AA)
            for idx,reverse in enumerate(self.reverse_list):
                cv2.putText(frame, "ID {} is reverse".format(self.reverse_list[-1]),
                            (500, 100 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No reverse", (100, 100+(idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, cv2.LINE_AA)


    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
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
