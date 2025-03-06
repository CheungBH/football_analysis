import cv2
from .utils import is_in_rectangle

class DelayRestartChecker:

    def __init__(self, court, fps=10, display_x=100, **kwargs):
        self.name = 'delay_restart'
        self.display_x = display_x
        self.field = court
        self.ball_coords = []
        self.flag = False
        self.flag_list=[]
        self.thre = 0.8
        self.counting_time = 10
        self.fps = fps
        self.last_ball = False
        self.whole_duration = self.counting_time * self.fps
        self.court = [(100, 180), (1000, 650)]


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

    def visualize(self, frame, idx):
        if self.flag == True:
            cv2.putText(frame, f'Delay restart', (self.display_x, 100+(idx*40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, f'No Delay restart', (self.display_x, 100+(idx*40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def visualize_details(self, frame, idx):
        self.visualize(frame, idx)
        pass


if __name__ == '__main__':
    import time
