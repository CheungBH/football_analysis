import cv2

class SideRefereeChecker:
    def __init__(self, court, **kwargs):
        self.flag = 0
        self.side_referee_coords = []
        self.court = court

    def process(self, side_referees,frame_id, **kwargs):
        court_line_left = 70
        court_line_top = 34
        court_line_bottom = 1171
        court_line_right = 1840

        for key, value_list in side_referees.items():
            # 遍历每个键对应的list
            for item in value_list:
                # 检查最后一个数字是否等于frame_id
                if item[-1] == frame_id:
                    if (item[0][1] < court_line_bottom and item[0][1] > court_line_top and item[0][0] > court_line_left and  item[0][0] < court_line_right):
                        self.flag = 1
                    else:
                        self.flag = 0
                else:
                    self.flag = -1
    def visualize(self, frame):
        if self.flag == 1:
            cv2.putText(frame, f"referee in court", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif self.flag == 0:
            cv2.putText(frame, f"referee out of court", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif self.flag == -1:
            cv2.putText(frame, f"No referee", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    def visualize_details(self, frame):
        self.visualize(frame)
        pass




