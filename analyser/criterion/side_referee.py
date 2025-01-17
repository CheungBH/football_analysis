import cv2

class SideRefereeChecker:
    def __init__(self, court, **kwargs):
        self.flag = False
        self.side_referee_coords = []
        self.court = court

    def process(self, side_referees, **kwargs):
        court_line_left = 70
        court_line_top = 34
        court_line_bottom = 1171
        court_line_right = 1840
        self.flag = False
        if len(side_referees):
            for k, v in side_referees.items():
                side_referees = v
            if (side_referees[-1][1] < court_line_bottom and
                    side_referees[-1][1] > court_line_top and
                    side_referees[-1][0] > court_line_left and
                    side_referees[-1][0] < court_line_right):
                self.flag = True
            else:
                self.flag = False

    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f"referee in court", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"referee out of court", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
        pass




