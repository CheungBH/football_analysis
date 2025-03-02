import cv2
from .utils import is_in_rectangle

class DelayRestartChecker:

    def __init__(self, court, fps=10, **kwargs):
        self.name = 'delay_restart'

        self.field = court
        self.ball_coords = []
        self.flag = False
        self.flag_list=[]
        self.thre = 0.8
        self.counting_time = 10
        self.fps = fps
        self.whole_duration = self.counting_time * self.fps
        self.court = [(100, 100), (1000, 650)]


    def process(self, balls, frame_queue, **kwargs):
        if balls:
            for ball in balls:
                if is_in_rectangle(ball, self.court):
                    ball = ball
            if is_in_rectangle(ball, self.court):
                self.flag_list.append(True)
            else:
                self.flag_list.append(False)
        else:
            self.flag_list.append(False)


        if len(self.flag_list) >= self.whole_duration:
            if sum(self.flag_list[-self.whole_duration:]) <= self.whole_duration*(1-self.thre):
                self.flag = True
            else:
                self.flag = False

    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'Delay restart', (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, f'No Delay restart', (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass


if __name__ == '__main__':
    import time
