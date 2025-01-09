
class SideRefereeChecker:
    def __init__(self, court):
        self.flag = False
        self.court = court

    def process(self, side_referee_coord, **kwargs):
        if side_referee_coord is not None:
            self.flag

    def visualize(self, frame):
        pass

    def visualize_details(self, frame):
        self.visualize(frame)
        pass




