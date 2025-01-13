from networkx.algorithms.bipartite.basic import color
from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.team_reverse_color = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=3, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_color_frame(self, frame):
        image = frame
        # cv2.imwrite(f"output/frame_{frame_id}_bbox_{bbox[0]}.jpg",image)
        # top_half_image = image[0:int(image.shape[0] / 2), :]
        top_half_image = image[int(image.shape[0] / 4):int(image.shape[0] / 2) , int(image.shape[1] / 4):int(3 * image.shape[1] / 4)]
        cv2.imshow('image', cv2.resize(top_half_image, (300, 300)))
        # cv2.imwrite(f"output_half/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

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
        # non_player_color1 =kmeans.cluster_centers_[0]
        # non_player_color2 = kmeans.cluster_centers_[1]
        # # non_player_color3 = kmeans.cluster_centers_[2]
        # # non_player_color4 = kmeans.cluster_centers_[3]
        # # non_player_color5 = kmeans.cluster_centers_[4]
        # player_color = kmeans.cluster_centers_[2]
        # return player_color,non_player_color1,non_player_color2

    def is_within_range(self,hsv, lower, upper):
        return all(lower[i] <= hsv[i] <= upper[i] for i in range(3))

    def get_player_color(self, frame, bbox, frame_id):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.imwrite(f"output1/frame_{frame_id}_bbox_{bbox[0]}.jpg",image)
        top_half_image = image[int(image.shape[0] / 8): int(image.shape[0] / 2),
                         int(image.shape[1] / 4):int(3 * image.shape[1] / 4)]
        # cv2.imshow('image', cv2.resize(top_half_image, (300, 300)))


        # cv2.imwrite(f"output1_half/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)
        colors = [kmeans.cluster_centers_[i] for i in range(3)]
        hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0].tolist() for color in colors]
        full_img = np.zeros((600, 900, 3), np.uint8)
        for i, color in enumerate(colors):
            full_img[0:300, i * 300:(i + 1) * 300] = color
            cv2.putText(full_img, f'H:{hsv_colors[i][0]} S:{hsv_colors[i][1]} V:{hsv_colors[i][2]}', (i * 300 + 50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('full-image', full_img)


        RED_LOWER_1 = [0, 120, 120]
        RED_UPPER_1 = [10, 255, 255]
        RED_LOWER_2 = [156, 120, 120]
        RED_UPPER_2 = [180, 255, 255]
        BLUE_LOWER = [100, 100, 120]
        BLUE_UPPER = [124, 255, 255]
        BLACK_LOWER = [0, 0, 0]
        BLACK_UPPER = [180, 255, 46]


        has_red = any(self.is_within_range(hsv, RED_LOWER_1, RED_UPPER_1) or self.is_within_range(hsv, RED_LOWER_2, RED_UPPER_2) for hsv in hsv_colors)
        has_blue = any(self.is_within_range(hsv, BLUE_LOWER, BLUE_UPPER) for hsv in hsv_colors)
        has_black = any(self.is_within_range(hsv, BLACK_LOWER, BLACK_UPPER) for hsv in hsv_colors)
        for hsv in hsv_colors:
            if has_red:
                player_color = np.array([0, 0, 255],dtype=np.uint8)
                print('red')
                # cv2.imwrite(f"output1/output_red/frame_{frame_id}_bbox_{bbox[0]}.jpg", top_half_image)
            elif has_blue:
                player_color = np.array([255, 0, 0],dtype=np.uint8)
                print('blue')
                # cv2.imwrite(f"output1/output_blue/frame_{frame_id}_bbox_{bbox[0]}.jpg", top_half_image)
            #elif (hsv[0]>=0 and hsv[0]<=180) and (hsv[1] >= 0 and hsv[1] <= 43) and (hsv[2] >= 46 and hsv[2] <= 220):
            elif has_black:
                player_color = np.array([0, 0, 0],dtype=np.uint8)
                print('black')
                # cv2.imwrite(f"output1/output_black/frame_{frame_id}_bbox_{bbox[0]}.jpg", top_half_image)
            else:
                player_color = np.array([125, 125, 125],dtype=np.uint8)
                print('gray')
                # cv2.imwrite(f"output1/output_gray/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
        # cv2.waitKey(0)

            # elif (hsv[0]>=0 and hsv[0]<=180) and (hsv[1] >= 0 and hsv[1] <= 255) and (hsv[2] >= 0 and hsv[2] <= 46):
            #     cv2.imwrite(f"output1/output_black/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0],
                           clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        non_player_color = kmeans.cluster_centers_[non_player_cluster]
        #player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_color(self):

        #self.team_colors = {0:np.array([0,0,255]), 1:np.array([125,125,125]), 2:np.array([255,0,0]), 3: np.array([0,0,0])}
        self.team_colors = {0:np.array([0,0,255],dtype=np.uint8), 1:np.array([125,125,125],dtype=np.uint8),
                            2:np.array([255,0,0],dtype=np.uint8), 3: np.array([0,0,0],dtype=np.uint8)}
        self.team_reverse_color = {tuple(value): key for key, value in self.team_colors.items()}
        '''
        classes_num = len(colors)
        # player_colors = [color1, color2]
        self.kmeans = KMeans(n_clusters=classes_num, init="k-means++", n_init=10)
        self.kmeans.fit(colors)
        for i in range(classes_num):
            self.team_colors[i] = self.kmeans.cluster_centers_[i]
        '''


    def get_player_team_test(self, frame, playerbbox, frame_id):
        # if self.kmeans is None:
        #     raise ValueError("Team colors have not been assigned yet. Call assign_team_color_test first.")

        player_color = self.get_player_color(frame, playerbbox, frame_id)
        team_id = self.team_reverse_color.get(tuple(player_color), None)

        #team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        return team_id

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

