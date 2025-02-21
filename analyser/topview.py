from collections import defaultdict
import cv2


class TopViewGenerator:
    def __init__(self, area_bounds):
        self.area_bounds = area_bounds
        self.points = []

    def merge_points_in_fixed_area(self, points, window_size):
        min_x, min_y, max_x, max_y = self.area_bounds
        grid = defaultdict(list)

        # Categorize points into grid cells within the specified area
        for x, y, team, color in points:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                grid_x = (x - min_x) // window_size  # Shift to start from min_x
                grid_y = (y - min_y) // window_size  # Shift to start from min_y
                grid[(grid_x, grid_y)].append((x, y, team, color))

        # Compute centroids for each grid cell
        merged_points = []
        for (grid_x, grid_y), cell_points in grid.items():
            if cell_points:
                avg_x = sum(p[0] for p in cell_points) / len(cell_points)
                avg_y = sum(p[1] for p in cell_points) / len(cell_points)
                t = cell_points[0][2]
                color = cell_points[0][3]
                merged_points.append((avg_x, avg_y, t, color))

        return merged_points

    def merge_points_same_team(self, points, window_size):
        min_x, min_y, max_x, max_y = self.area_bounds
        grid = defaultdict(list)

        # Categorize points into grid cells within the specified area
        for x, y, team, color in points:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                grid_x = (x - min_x) // window_size  # Shift to start from min_x
                grid_y = (y - min_y) // window_size  # Shift to start from min_y
                grid[(grid_x, grid_y)].append((x, y, team, color))

        # Compute centroids for each grid cell
        merged_points = []
        for (grid_x, grid_y), cell_points in grid.items():
            teams = set(p[2] for p in cell_points)
            if len(teams) == 1 and cell_points:
                avg_x = sum(p[0] for p in cell_points) / len(cell_points)
                avg_y = sum(p[1] for p in cell_points) / len(cell_points)
                t = cell_points[0][2]
                color = cell_points[0][3]
                merged_points.append((avg_x, avg_y, t, color))
            else:
                # If there are multiple teams in the same grid cell, ignore the cell
                for cell_point in cell_points:
                    merged_points.append(cell_point)

        return merged_points

    def remove_out_ball(self, ball_points):
        ball_points = [ball for ball in ball_points if ball[0] > 50 and ball[0] < 1100 and ball[1] > 50 and ball[1] < 720]
        return ball_points

    def process(self, player_points, ball_points):
        self.all_player_points = player_points
        self.all_ball_points = ball_points
        player_points = self.merge_points_same_team(player_points, 10)
        player_points = self.merge_points_in_fixed_area(player_points, 100)
        self.player_points = player_points
        ball_points = self.remove_out_ball(ball_points)
        self.ball_points = ball_points

    def visualize(self, top_view_img):
        for player in self.player_points:
            cv2.circle(top_view_img, (int(player[0]), int(player[1])), 20, tuple(player[3]), -1)
        for ball in self.ball_points:
            cv2.circle(top_view_img, (int(ball[0]), int(ball[1])), 20, (0, 255, 0), -1)
