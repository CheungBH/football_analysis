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

        '''
        defaultdict(<class 'list'>, {7: [[[165.13565067217837, 957.2850765398875], 0], [[165.2379354575891, 957.0820731218867], 1]],59: [[[168.44876559009623, 955.0804552127336], 553], [[177.66906955076163, 948.9450571785179], 572]]})    
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
        '''

    def visualize(self, frame):
        if self.flag == 1:
            cv2.putText(frame, f"referee in court", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        elif self.flag == 0:
            cv2.putText(frame, f"referee out of court", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        elif self.flag == -1:
            cv2.putText(frame, f"No referee", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    def visualize_details(self, frame):
        self.visualize(frame)
        pass




