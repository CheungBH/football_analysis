import cv2


class BallOutRangeChecker:
    def __init__(self, court, **kwargs):
        self.field = court
        self.ball_coords = []
        self.flag = False #True visualize

    def process(self, balls, **kwargs):
        # self.ball_coords.append(balls)
        court_line_left=83
        court_line_top=34
        court_line_bottom=1171
        if len(balls):
            if balls[-1][0] < court_line_left or balls[-1][1] < court_line_top or balls[-1][1]>court_line_bottom:
                self.flag = True
            else:
                self.flag = False

    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'out of court', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'In court', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass


if __name__ == '__main__':
    import time