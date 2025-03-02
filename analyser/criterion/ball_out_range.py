import cv2
from .utils import is_in_rectangle

class BallOutRangeChecker:

    def __init__(self, court, **kwargs):
        self.name = 'ball_out_range'

        self.field = court
        self.ball_coords = []
        self.flag = False
        self.flag_list=[]
        self.thre = 0.8
        self.frame_duration = 10

    def process(self, balls,frame_queue, **kwargs):
        court = [(50, 50), (1100, 730)]
        if balls:
            for ball in balls:
                if is_in_rectangle(ball,court):
                    ball = ball
            if is_in_rectangle(ball,court):
                flag = True
            else:
                flag = False
        else:
            flag = False

        self.flag_list.append(flag)
        if len(self.flag_list) >= self.frame_duration:
            if sum(self.flag_list[-self.frame_duration:]) <= self.frame_duration*(1-self.thre):
                self.flag = True
            else:
                self.flag = False
        #
        # if len(self.flag_list) >= self.frame_duration:
        #     filter_flag_list = [item for item in self.flag_list[-self.frame_duration:] if item is not None]
        #     if sum(filter_flag_list) > (self.frame_duration * self.thre):
        #         self.flag = True
        #     else:
        #         self.flag = False

    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'Ball out of court', (100, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, f'Ball in court', (100, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass


if __name__ == '__main__':
    import time
