
class BallOutRangeChecker:
    def __init__(self, field):
        self.field = field
        self.ball_coords = []
        self.flag = False

    def process(self, ball_coord, **kwargs):
        self.ball_coords.append(ball_coord)
        if self.ball_coords:
            pass

    def visualize(self, frame):
        pass

    def visualize_details(self, frame):
        self.visualize(frame)
        pass

