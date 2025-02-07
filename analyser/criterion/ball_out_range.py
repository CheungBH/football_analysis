import cv2


class BallOutRangeChecker:
    def __init__(self, court, **kwargs):
        self.field = court
        self.ball_coords = []
        self.flag = False #True visualize
        self.flag_list=[]

    def process(self, balls, **kwargs):
        # self.ball_coords.append(balls)
        court_line_left=83
        court_line_top=34
        court_line_bottom=1171
        if len(balls):
            if balls[-1][0] < court_line_left or balls[-1][1] < court_line_top or balls[-1][1]>court_line_bottom:
                self.flag_list.append(True)
            else:
                self.flag_list.append(False)

    def visualize(self, frame):

        if len(self.flag_list)>5:
            if sum(self.flag_list[-5:]) >4:
                cv2.putText(frame, f'out of court', (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                self.flag = True
            else:
                cv2.putText(frame, f'In court', (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                self.flag = False

    def visualize_details(self, frame):
        self.visualize(frame)
        pass


if __name__ == '__main__':
    import time
