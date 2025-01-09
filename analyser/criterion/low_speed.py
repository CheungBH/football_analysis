import cv2
from .utils import check_speed_ls

class LowSpeedChecker:
    def __init__(self):
        self.speed_threshold = 1000
        self.frame_duration = 10
        self.flag = False

    def process(self, candidates):
        candidates = self.preprocess(candidates)
        self.status = []
        for tid, candidate in candidates.items():
            if len(candidate) < self.frame_duration:
                self.status.append(False)
            else:
                speed = check_speed_ls(candidate)
                if speed < self.speed_threshold:
                    self.status.append(False)
                else:
                    self.status.append(True)
        return self.status

    def visualize(self, frame):
        if not self.status[-1]:
            cv2.putText(frame, "Normal speed", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Low Speed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        2, cv2.LINE_AA)

    def visualize_details(self, frame):
        self.visualize(frame)
