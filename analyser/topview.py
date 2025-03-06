import copy
import os
from collections import defaultdict
import cv2
import random
import numpy as np
# from networkx.algorithms.centrality import voterank


class TopViewGenerator:
    def __init__(self, area_bounds):
        self.area_bounds = area_bounds
        self.whole_bounds = [0, 0, 1200, 800]
        self.points = []
        self.num_dict = {0:11, 1:11, 2:1, 3:1, 4:1}
        self.select_top = [[2,3,4], [0,10000,575]]

    def save_topview_img(self, top_view_img, players, balls, frame_idx, path):
        for player in players:
            cv2.circle(top_view_img, (int(player[0]), int(player[1])), 20, tuple(player[3]), -1)
            cv2.putText(top_view_img, str(player[-1]),(int(player[0]), int(player[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        for ball in balls:
            cv2.circle(top_view_img, (int(ball[0]), int(ball[1])), 20, (0, 255, 0), -1)
            cv2.circle(top_view_img, (int(ball[0]), int(ball[1])), 150, (0, 0, 255), 2)
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f'tv_{frame_idx}.jpg'), top_view_img)

    def constrain_number_of_points(self, points, all_points, idx):
        number = self.num_dict[idx]
        if len(points) == number:
            return points
        elif len(points) > number:
            return random.sample(points, number)
        else:
            if len(all_points) <= number:
                return all_points
            sample_point = random.sample(all_points, number - len(points))
            return points + sample_point

    def keep_distance_points(self, points, team, dist):
        team_points = [point for point in points if point[2] == team]
        valid_points = []
        y_area_bounds = [self.area_bounds[1], self.area_bounds[3]]
        for team_point in team_points:
            y_coord = team_point[1]
            for y_bound in y_area_bounds:
                if abs(y_coord - y_bound) < dist:
                    valid_points.append(team_point)
                    continue
        other_points = [point for point in points if point not in valid_points]
        return valid_points, other_points


    def player_stable(self, player_points, player_all_points):
        points_dict = defaultdict(list)
        point_all_dict = defaultdict(list)
        for idx in range(5):
            for point in player_points:
                if point[2] == idx:
                    points_dict[idx].append(point)

            for all_point in player_all_points:
                if all_point[2] == idx:
                    point_all_dict[idx].append(all_point)

        final_points = []

        for idx, points in points_dict.items():
            final_points += self.constrain_number_of_points(points_dict[idx], point_all_dict[idx], idx)
        return final_points


    def save_tmp_videos(self, folder, writer, size):
        tv_imgs = [cv2.imread(os.path.join(folder, "tv_{}.jpg".format(i))) for i in range(4)]
        top_row = np.hstack([tv_imgs[1], tv_imgs[0]])
        bottom_row = np.hstack((tv_imgs[3], tv_imgs[2]))
        combined_frame = np.vstack([top_row, bottom_row])
        combined_frame = cv2.resize(combined_frame, size)
        writer.write(combined_frame)
        return combined_frame

    def merge_points_in_fixed_area(self, points, window_size):
        min_x, min_y, max_x, max_y = self.area_bounds
        grid = defaultdict(list)

        # Categorize points into grid cells within the specified area
        for x, y, team, color, idx in points:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                grid_x = (x - min_x) // window_size  # Shift to start from min_x
                grid_y = (y - min_y) // window_size  # Shift to start from min_y
                grid[(grid_x, grid_y)].append((x, y, team, color, idx))

        # Compute centroids for each grid cell
        merged_points = []
        for (grid_x, grid_y), cell_points in grid.items():
            t_all = set(p[2] for p in cell_points)
            if cell_points:
                avg_x = sum(p[0] for p in cell_points) / len(cell_points)
                avg_y = sum(p[1] for p in cell_points) / len(cell_points)
                # t = cell_points[0][2]
                t = 4 if 4 in t_all else  cell_points[0][2]
                color = cell_points[0][3]
                idx = cell_points[0][4]
                merged_points.append((avg_x, avg_y, t, color, idx))

        return merged_points

    def merge_points_same_team(self, points, window_size):
        min_x, min_y, max_x, max_y = self.whole_bounds
        grid = defaultdict(list)

        # Categorize points into grid cells within the specified area
        for x, y, team, color,idx in points:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                grid_x = (x - min_x) // window_size  # Shift to start from min_x
                grid_y = (y - min_y) // window_size  # Shift to start from min_y
                grid[(grid_x, grid_y)].append((x, y, team, color, idx))

        # Compute centroids for each grid cell
        merged_points = []
        for (grid_x, grid_y), cell_points in grid.items():
            teams = set(p[2] for p in cell_points)
            if len(teams) == 1 and cell_points:
                avg_x = sum(p[0] for p in cell_points) / len(cell_points)
                avg_y = sum(p[1] for p in cell_points) / len(cell_points)
                t = cell_points[0][2]
                color = cell_points[0][3]
                idx = cell_points[0][4]
                merged_points.append((avg_x, avg_y, t, color, idx))
            else:
                # If there are multiple teams in the same grid cell, ignore the cell
                for cell_point in cell_points:
                    merged_points.append(cell_point)

        return merged_points

    def remove_out_ball(self, ball_points):
        ball_points = [ball for ball in ball_points if ball[0] > 50 and ball[0] < 1100 and ball[1] > 50 and ball[1] < 720]
        ball_points = [ball_points[0]] if len(ball_points) > 0 else []
        return ball_points

    def cal_dist(self, point1, point2):
        return abs(point1 - point2)

    def select_top_points(self, points, teams):
        select_team = self.select_top[0]
        dist = self.select_top[1]
        top_candidates = [point for point in points if point[2] in teams]
        others_candidates = [point for point in points if point[2] not in teams]
        # for candidate in top_candidates:
        selected_points = []
        for st, d in zip(select_team, dist):
            if st not in teams:
                continue
            team_candidates = [point for point in top_candidates if point[2] == st]
            closest_idx = -1
            closest_dist = float('inf')
            for idx, candidate in enumerate(team_candidates):
                dist = self.cal_dist(candidate[0], d)
                if dist < closest_dist:
                    closest_idx = idx
                    closest_dist = dist

            if closest_idx == -1:
                return others_candidates
            selected_points.append(team_candidates[closest_idx])
        selected_points += others_candidates
        return selected_points


    def process(self, player_points, ball_points, top_view_img, save_tmp=""):
        self.all_player_points = player_points
        self.all_ball_points = ball_points

        if save_tmp:
            save_tmp = os.path.join(save_tmp, "merge_steps")
            os.makedirs(save_tmp, exist_ok=True)
            self.save_topview_img(copy.deepcopy(top_view_img), player_points, ball_points, "1_all_points", save_tmp)

        player_points = self.select_top_points(player_points, teams=[2, 3])
        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), player_points, ball_points, "2_removed_repeat_goalkeeper", save_tmp)
        player_points = self.merge_points_same_team(player_points, 20)
        merged_points = player_points

        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), player_points, ball_points, "3_merged_same_team_in_small_area", save_tmp)

        side_referee_points, player_points = self.keep_distance_points(player_points, 4, 30)
        player_points = self.select_top_points(player_points, teams=[4])
        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), player_points, ball_points, "4_Processed_referee", save_tmp)
        player_points = self.merge_points_in_fixed_area(player_points, 50)
        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), player_points, ball_points, "5_merged_points_in_bigger_area", save_tmp)
        player_points = self.player_stable(player_points, merged_points)
        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), player_points, ball_points, "6_stable_points", save_tmp)
        player_points += side_referee_points
        self.player_points = player_points
        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), self.player_points, ball_points, "7_final_without_ball", save_tmp)
        ball_points = self.remove_out_ball(ball_points)
        ball_points = ball_points[0:1]
        # ball_points = self.merge_points_in_fixed_area(ball_points, 30)
        self.ball_points = ball_points
        if save_tmp:
            self.save_topview_img(copy.deepcopy(top_view_img), self.player_points, self.ball_points, "FINAL", save_tmp)

    def visualize(self, top_view_img):
        for player in self.player_points:
            cv2.circle(top_view_img, (int(player[0]), int(player[1])), 20, tuple(player[3]), -1)
        for ball in self.ball_points:
            cv2.circle(top_view_img, (int(ball[0]), int(ball[1])), 20, (0, 255, 0), -1)
