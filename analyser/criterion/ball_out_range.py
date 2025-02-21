import cv2


class BallOutRangeChecker:
    def __init__(self, court, **kwargs):
        self.field = court
        self.ball_coords = []
        self.flag = False
        self.flag_list=[]
        self.ratio = 0.5

    def process(self, balls,frame_queue, **kwargs):
        # self.ball_coords.append(balls)
        court_line_left=50
        court_line_top=50
        court_line_bottom=1170
        court_line_right=720
        if len(balls):
            if balls[-1][0] < court_line_left or balls[-1][0] > court_line_right\
                    or balls[-1][1] < court_line_top or balls[-1][1]>court_line_bottom:
                self.flag_list.append(True)
            else:
                self.flag_list.append(False)
        if len(self.flag_list) > frame_queue:
            if sum(self.flag_list[-frame_queue:]) > (frame_queue * self.ratio):
                self.flag = True
            else:
                self.flag = False
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
