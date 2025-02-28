import cv2

class CommitFoulChecker:
    def __init__(self,court, **kwargs):
        self.name = 'commit_foul'
        self.frame_duration = 10
        self.flag = False
        self.flag_list = []
        self.thre = 0.8
        self.window_size = 100
        self.step_size = 50
        self.points_trigger = 8

    def check_points_in_window(self, points):

        if not points:
            return False

        # Get the bounds of the area to cover
        min_x = min(x for x, y in points)
        max_x = max(x for x, y in points)
        min_y = min(y for x, y in points)
        max_y = max(y for x, y in points)

        # Slide the window across the plane
        for x in range(int(min_x), int(max_x) + 1, int(self.step_size)):
            for y in range(int(min_y), int(max_y) + 1, int(self.step_size)):
                # Count points inside the current window
                count = sum(1 for px, py in points if x <= px < x + self.window_size and y <= py < y + self.window_size)
                if count > self.points_trigger:
                    return True

        return False

    def process(self, player_current, **kwargs):
        # Extract all player points
        player_points = [(x, y) for points in player_current.values() if isinstance(points, list) and len(points) == 2 for x, y in [points]]
        #player_points = [(x, y) for points in player_current.values() if isinstance(points, (list, tuple)) for x, y in points]

        # Check if a foul was committed
        self.flag_list.append(self.check_points_in_window(player_points))

        # Update the flag
        if len(self.flag_list) >= self.frame_duration:
            if sum(self.flag_list[-self.frame_duration:]) > (self.frame_duration * self.thre):
                self.flag = True
            else:
                self.flag = False

    def visualize(self, frame):
        if self.flag == True:
            cv2.putText(frame, f'Commit Foul', (100, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, f'No Foul', (100, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    def visualize_details(self, frame):
        self.visualize(frame)
        pass
