import cv2
from .utils import is_point_in_triangle, is_in_rectangle


class GoalKeeperPoorPositioningChecker:
    def __init__(self, **kwargs):
        self.frame_duration = 10
        self.flag = False
        self.flag_list = []
        self.thre = 0.6
        # self.restricted_area = [(50, 320), (215, 476)]

    def detect_satisfy_condition(self, players, balls):
        rect1 = [(50, 320), (215, 476)]
        rect2 = [(960, 320), (1200, 586)]
        return False
        if not balls:
            return False
        ball = None
        for b in balls:
            if is_in_rectangle(b, rect1[0], rect1[1], rect2[1]):
                ball = b
                break
        # if less than 3 players inside the restricted area, the goalkeeper will be the closet one to door
        count = 0
        return


    def process(self, players, balls,frame_queue, **kwargs):
        if not self.detect_satisfy_condition(players, balls):
            self.flag_list.append(False)

        goal_keeper, door_upper, door_lower, ball = (0, 0), (0, 0), (0, 0), (0, 0)
        # Need to detect
        if is_point_in_triangle(goal_keeper, door_upper, door_lower, ball):
            self.flag_list.append(True)
        else:
            self.flag_list.append(False)

        if len(self.flag_list) >= self.frame_duration:
            if sum(self.flag_list[-self.frame_duration:]) <= self.frame_duration*(1-self.thre):
                self.flag = True
            else:
                self.flag = False


    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, "Goalkeeper Condition poor", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Goalkeeper Condition OK", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, cv2.LINE_AA)

    def visualize_details(self):
        self.visualize()

