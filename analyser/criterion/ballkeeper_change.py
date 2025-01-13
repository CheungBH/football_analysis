from collections import defaultdict
import math
import cv2
class BallKeeperChangeChecker:
    def __init__(self, max_change_thresh=10):
        self.max_change_thresh = max_change_thresh
        self.flag = False
        self.catcher = 1
        self.first_catcher = 1
        self.catch_list = []
        self.thre = 5
        self.holder = 0


    def calculate_distances(self, points_dict, ref_point, item):
        distances = {}
        for key, value in points_dict.items():
            point = value[item]
            distance = math.sqrt((point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2)
            distances[key] = distance
        return distances

    def find_closest_player(self, players, ball_position, item):
        distances = self.calculate_distances(players, ball_position, item)
        min_key = min(distances, key=distances.get)
        return min_key, distances[min_key]

    def check_ball_possession(self,lst):
        count_1 = 0
        count_2 = 0 # 持球状态：0-交错，1-1持球，2-2持球
        prev_possession,possession =0
        possession_list=[]
        possession_changed = False
        for i in range(len(lst)):
            if lst[i] == 1:
                count_1 += 1
                count_2 = 0
                if count_1 >= self.thre:
                    possession = 1
                    possession_list.append(possession)
            elif lst[i] == 2:
                count_2 += 1
                count_1 = 0
                if count_2 >= self.thre:
                    possession = 2
                    possession_list.append(possession)
            else:
                count_1 = 0
                count_2 = 0
                possession = 0

            # 如果 1 和 2 交错，判定为 0
            if i > 0 and lst[i] != lst[i - 1]:
                count_1 = 0
                count_2 = 0
                possession = 0
            if len(possession_list) > 1:
                if possession_list[-1] != possession_list[-2]:
                    possession_changed = True

        return possession,possession_changed

    def process(self, players1, players2, balls, **kwargs):
        ball_position = balls[-1]

        # 找到第一次接球者
        min_key_raw1, min_distance_raw1 = self.find_closest_player(players1, ball_position, 0)
        min_key_raw2, min_distance_raw2 = self.find_closest_player(players2, ball_position, 0)
        self.first_catcher = 1 if min_distance_raw1 < min_distance_raw2 else 2


        # 找到最终接球者
        min_key_1, min_distance_1 = self.find_closest_player(players1, ball_position, -1)
        min_key_2, min_distance_2 = self.find_closest_player(players2, ball_position, -1)
        self.catcher = 1 if min_distance_1 < min_distance_2 else 2
        self.catch_list.append(self.catcher)
        if len(self.catch_list)>5:
            self.holder,self.flag = self.check_ball_possession(self.catch_list)

    def visualize(self, frame):
        if self.holder == 1:
            cv2.putText(frame, f'Team1 catch the ball', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        elif self.holder == 2:
            cv2.putText(frame, f'Team2 catch the ball', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        elif self.holder == 0:
            cv2.putText(frame, f'Fighting for the ball', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if self.flag == True:
            cv2.putText(frame, f'Ball change', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass

