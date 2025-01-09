import cv2


class MovingReverseChecker:
    def __init__(self):
        self.th

    def process(self, balls, players1, players2, **kwargs):
        pass

    def visualize(self, frame):
        if not self.status[-1]:
            cv2.putText(frame, "Normal speed", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Low Speed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
