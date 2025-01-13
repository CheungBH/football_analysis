import cv2


class BallOutRangeChecker:
    def __init__(self, field):
        self.field = field
        self.ball_coords = []
        self.flag = False #True visualize

    def process(self, balls, **kwargs):
        # self.ball_coords.append(balls)
        if balls[-1][0] < 83 or balls[-1][1]<34 or balls[-1][1]>1171:
            self.flag = True


    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'out of court', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'In court', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


    def visualize_details(self, frame):
        self.visualize(frame)
        pass

