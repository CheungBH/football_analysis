# from networkx.algorithms.bipartite.basic import color
from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self, **kwargs):
        self.team_colors = {}
        self.team_reverse_color = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.green_idx = 2
        self.larger_criterion = {0: 1}

    def get_clustering_model(self, image,n_clusters):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_color_frame(self, frame):
        image = frame
        # cv2.imwrite(f"output/frame_{frame_id}_bbox_{bbox[0]}.jpg",image)
        # top_half_image = image[0:int(image.shape[0] / 2), :]
        top_half_image = image[int(image.shape[0] / 6):int(image.shape[0] / 2) , int(image.shape[1] / 4):int(3 * image.shape[1] / 4)]
        cv2.imshow('image', cv2.resize(top_half_image, (300, 300)))
        # cv2.imwrite(f"output_half/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image,6)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        '''
        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0],
                           clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        non_player_color =kmeans.cluster_centers_[non_player_cluster]
        # non_player_color2 = kmeans.cluster_centers_[non_player_cluster]
        # non_player_color3 = kmeans.cluster_centers_[non_player_cluster]
        player_color = kmeans.cluster_centers_[player_cluster]
        '''

        colors = [kmeans.cluster_centers_[i] for i in range(3)]
        return colors


    def is_within_range(self,hsv, lower, upper):
        return all(lower[i] <= hsv[i] <= upper[i] for i in range(3))

    def get_first_pixel_colors(self,folder_path):
        team_color = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image = cv2.imread(os.path.join(folder_path, file_name))
                first_pixel_color = image[0, 0]
                team_color.append(first_pixel_color.tolist())
        return team_color

    def filter_colors(self,clustered_colors, team_colors, threshold=80):
        filtered_colors_raw = []
        filtered_colors = []
        indices = []
        similarities = []

        for index1, color1 in enumerate(clustered_colors):
            min_similarity = float('inf')
            closest_color = None

            for index2,color2 in enumerate(team_colors):
                similarity = np.linalg.norm(np.array(color1) - np.array(color2))
                if similarity < min_similarity:
                    min_similarity = similarity
                    closest_color = color2
                    idx = index2

            if min_similarity < threshold:
                filtered_colors_raw.append([int(c) for c in color1.tolist()])
                filtered_colors.append(closest_color)
                similarities.append(min_similarity)
                indices.append(idx)

        return filtered_colors_raw, filtered_colors, indices, similarities
    def get_player_color(self, frame, bbox, frame_id,team_colors):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # cv2.imshow('image', cv2.resize(top_half_image, (300, 300)))
        # cv2.imwrite(f"output1/frame_{frame_id}_bbox_{bbox[0]}.jpg",image)
        top_half_image = image[int(image.shape[0] / 6): int(image.shape[0] / 2),
                         0:int( image.shape[1])]
        image_2d = top_half_image.reshape(-1,3)
        team_colors = list(team_colors.values())
        initial_centers1 = np.array(team_colors)
        kmeans1 = KMeans(n_clusters=len(initial_centers1), init=initial_centers1, n_init=1)
        kmeans1.fit(image_2d)
        cluster_centers = kmeans1.cluster_centers_ # 聚类颜色
        filtered_colors_raw, filtered_colors, indices, similarity = self.filter_colors(cluster_centers, team_colors,80) # caculate_distance,min_distance

        if similarity ==[]:
            player_color = np.array([255, 0, 0],dtype=np.uint8)
        else:
            index_list=[]
            labels = kmeans1.labels_ # pixel2
            for index,label in enumerate(labels):
                if label in indices:
                    index_list.append(index)
            pixels = [image_2d[idx] for idx in index_list]

            filtered_team_color = list(set([tuple(color) for color in filtered_colors]))
            initial_centers2 = np.array(filtered_team_color)
            kmeans2 = KMeans(n_clusters=len(initial_centers2), init=initial_centers2, n_init=1)
            kmeans2.fit(pixels)
            color2 = kmeans2.cluster_centers_
            filtered_colors_raw2, filtered_colors2, indices2, similarity2 = self.filter_colors(color2, initial_centers2,99999)

            labels_2 = kmeans2.predict(pixels)
            label_counts_2 = np.bincount(labels_2)
            total_samples2 = len(labels_2)
            ratios_2 = label_counts_2 / total_samples2

            is_green = []
            for idx,color2 in enumerate(filtered_colors2):
                if color2.tolist() == self.team_colors[2]:
                    is_green.append(True)
                else:
                    is_green.append(False)
                if similarity2[idx] == min(similarity2):
                    player_color = color2

            # player_color = np.array([0,0,0])
            for idx in range(len(filtered_colors2)):
                if is_green[idx] and ratios_2[idx] > 0.93:
                    player_color = filtered_colors2[idx]
                    return player_color

            max_ratio = float("-inf")
            for idx in range(len(filtered_colors2)):
                if not is_green:
                    if ratios_2[idx] > max_ratio:
                        player_color = filtered_colors2[idx]
            return player_color


            full_img = np.zeros((500, 700, 3), np.uint8)
            resized_img = cv2.resize(image, (500, 500))
            full_img[0:500, 0:500] = resized_img
            for i, color in enumerate(filtered_colors_raw2):
                full_img[i * 100:(i + 1) * 100, 500:600] = color
                full_img[i * 100:(i + 1) * 100, 600:700] = filtered_colors2[i]
                cv2.putText(full_img, 'similarity:{}'.format(similarity2[i]), (i * 300 + 50, 550),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.imshow('Result', full_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()


        # n_cluster = 6
        # kmeans = self.get_clustering_model(top_half_image,n_cluster)
        # colors = [kmeans.cluster_centers_[i] for i in range(n_cluster)]
        # hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0].tolist() for color in colors]
        #
        # labels = kmeans.labels_
        # label_counts = np.bincount(labels)
        # total_samples = len(labels)
        # weights = label_counts / total_samples
        # color_groups = defaultdict(list)
        # for color in colors:
        #     found_group = False
        #     for group_color in color_groups:
        #         if np.linalg.norm(color - np.array(group_color)) <= 100:
        #             color_groups[group_color].append(color)
        #             found_group = True
        #             break
        #     if not found_group:
        #         color_groups[tuple(color)].append(color)
        # # 找到出现最多的颜色类别
        # dominant_group = max(color_groups.values(), key=len)
        # dominant_color = np.mean(dominant_group, axis=0)
        #
        #
        # full_img = np.zeros((900, 900, 3), np.uint8)
        # for i, color in enumerate(colors):
        #     if i < 3:
        #         full_img[0:300, i * 300:(i + 1) * 300] = color  # 第一行
        #         cv2.putText(full_img, f'H:{hsv_colors[i][0]} S:{hsv_colors[i][1]} V:{hsv_colors[i][2]}', (i * 300 + 50, 150),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        #     elif i>=3 and i<6:
        #         full_img[300:600, (i - 3) * 300:(i - 2) * 300] = color  # 第二行
        #         cv2.putText(full_img, f'H:{hsv_colors[i][0]} S:{hsv_colors[i][1]} V:{hsv_colors[i][2]}', ((i - 3) * 300 + 50, 450),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        #     full_img[600:900, (i - 3) * 300:(i - 2) * 300] = dominant_color
        #     cv2.imshow('full-image', full_img)
        #
        #
        # RED_LOWER_1 = [0, 120, 120]
        # RED_UPPER_1 = [10, 255, 255]
        # RED_LOWER_2 = [156, 120, 120]
        # RED_UPPER_2 = [180, 255, 255]
        # BLUE_LOWER = [100, 100, 120]
        # BLUE_UPPER = [124, 255, 255]
        # BLACK_LOWER = [0, 0, 0]
        # BLACK_UPPER = [180, 255, 46]
        #
        #
        # has_red = any(self.is_within_range(hsv, RED_LOWER_1, RED_UPPER_1) or self.is_within_range(hsv, RED_LOWER_2, RED_UPPER_2) for hsv in hsv_colors)
        # has_blue = any(self.is_within_range(hsv, BLUE_LOWER, BLUE_UPPER) for hsv in hsv_colors)
        # has_black = any(self.is_within_range(hsv, BLACK_LOWER, BLACK_UPPER) for hsv in hsv_colors)
        # for hsv in hsv_colors:
        #     if has_red:
        #         player_color = np.array([0, 0, 255],dtype=np.uint8)
        #     elif has_blue:
        #         player_color = np.array([255, 0, 0],dtype=np.uint8)
        #     elif has_black:
        #         player_color = np.array([0, 0, 0],dtype=np.uint8)
        #     else:
        #         player_color = np.array([125, 125, 125],dtype=np.uint8)
        # # Get the cluster labels for each pixel
        # labels = kmeans.labels_
        #
        # # Reshape the labels to the image shape
        # clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        # # Get the player cluster
        # corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0],
        #                    clustered_image[-1, -1]]
        # non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        # player_cluster = 1 - non_player_cluster
        # non_player_color = kmeans.cluster_centers_[non_player_cluster]
        # player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_color(self,team_colors):

        #self.team_colors = {0:np.array([0,0,255]), 1:np.array([125,125,125]), 2:np.array([255,0,0]), 3: np.array([0,0,0])}

        self.team_colors = team_colors
        self.team_reverse_color = {tuple(value): key for key, value in self.team_colors.items()}
        '''
        classes_num = len(colors)
        # player_colors = [color1, color2]
        self.kmeans = KMeans(n_clusters=classes_num, init="k-means++", n_init=10)
        self.kmeans.fit(colors)
        for i in range(classes_num):
            self.team_colors[i] = self.kmeans.cluster_centers_[i]
        '''


    def get_player_team_test(self, frame, playerbbox, frame_id, team_colors):
        # if self.kmeans is None:
        #     raise ValueError("Team colors have not been assigned yet. Call assign_team_color_test first.")

        player_color = self.get_player_color(frame, playerbbox, frame_id,team_colors)
        team_id = self.team_reverse_color.get(tuple(player_color), None)

        #team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        return team_id if team_id is not None else 0

    def get_player_whole_team(self, frame, player_bboxs, player_id, team_colors, **kwargs):
        team_ids = []
        for player_bbox in player_bboxs:
            try:
                team_id = self.get_player_team_test(frame, player_bbox, player_id, team_colors)
            except:
                team_id = 0
            team_ids.append(team_id)
        return team_ids

    def get_player_team(self, frame, player_bbox, player_id):
        if self.kmeans is None:
            raise ValueError("Team colors have not been assigned yet. Call assign_team_color first.")

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id


if __name__ == '__main__':
    import os
    from pyreadline3.console import BLACK

    assigner = TeamAssigner()
    image_folder = (r"C:\hku\program\football_analysis\output1\output_black")
    # H:red:118-122 blue:7-10
    img_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
    for img_path in img_paths:
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (800, 400))
        colors = assigner.get_color_frame(img)
        # player_color, non_player_color1, non_player_color2 = assigner.get_color_frame(img)

        hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0] for color in colors]
        for i, color in enumerate(hsv_colors):
            print(f'Color {i} - H: {color[0]}, V: {color[2]}')


        # 创建纯色图像
        color_imgs = []
        for color in colors:
            img = np.zeros((300, 300, 3), np.uint8)
            img[:] = color
            color_imgs.append(img)

        full_img = np.zeros((600, 900, 3), np.uint8)
        for i, color in enumerate(colors):
            full_img[0:300, i * 300:(i + 1) * 300] = color
            cv2.putText(full_img, f'H:{hsv_colors[i][0]} V:{hsv_colors[i][2]}', (i * 300 + 50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(os.path.basename(img_path), resized_img)
        cv2.imshow('full-image', full_img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyWindow(os.path.basename(img_path))

