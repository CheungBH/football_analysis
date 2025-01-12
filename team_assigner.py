from networkx.algorithms.bipartite.basic import color

from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
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
        top_half_image = image[:]

        # cv2.imwrite(f"output1_half/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)
        colors = [kmeans.cluster_centers_[i] for i in range(3)]
        hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0].tolist() for color in colors]

        RED_LOWER_1 = [0, 43, 46]
        RED_UPPER_1 = [10, 255, 255]
        RED_LOWER_2 = [156, 43, 46]
        RED_UPPER_2 = [180, 255, 255]
        BLUE_LOWER = [100, 100, 46]
        BLUE_UPPER = [124, 255, 255]
        BLACK_LOWER = [0, 0, 0]
        BLACK_UPPER = [180, 255, 46]

        has_red = any(self.is_within_range(hsv, RED_LOWER_1, RED_UPPER_1) or self.is_within_range(hsv, RED_LOWER_2, RED_UPPER_2) for hsv in hsv_colors)
        has_blue = any(self.is_within_range(hsv, BLUE_LOWER, BLUE_UPPER) for hsv in hsv_colors)
        has_black = any(self.is_within_range(hsv, BLACK_LOWER, BLACK_UPPER) for hsv in hsv_colors)
        for hsv in hsv_colors:
            if has_red:
                player_color = np.array([0, 0, 255])
                #cv2.imwrite(f"output1/output_red/frame_{frame_id}_bbox_{bbox[0]}.jpg", top_half_image)
            elif has_blue:
                player_color = np.array([255, 0, 0])
                #cv2.imwrite(f"output1/output_blue/frame_{frame_id}_bbox_{bbox[0]}.jpg", top_half_image)
            #elif (hsv[0]>=0 and hsv[0]<=180) and (hsv[1] >= 0 and hsv[1] <= 43) and (hsv[2] >= 46 and hsv[2] <= 220):
            elif has_black:
                player_color = np.array([0, 0, 0])
                #cv2.imwrite(f"output1/output_black/frame_{frame_id}_bbox_{bbox[0]}.jpg}", top_half_image)
            else:
                player_color = np.array([125, 125, 125])
                #cv2.imwrite(f"output1/output_gray/frame_{frame_id}_bbox_{bbox[0]}_half.jpg", top_half_image)
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

    def assign_color(self, colors):
        classes_num = len(colors)
        # player_colors = [color1, color2]
        self.kmeans = KMeans(n_clusters=classes_num, init="k-means++", n_init=10)
        self.kmeans.fit(colors)

        for i in range(classes_num):
            self.team_colors[i] = self.kmeans.cluster_centers_[i]

    def get_player_team_test(self, frame, playerbbox, frame_id):
        # if self.kmeans is None:
        #     raise ValueError("Team colors have not been assigned yet. Call assign_team_color_test first.")

        player_color = self.get_player_color(frame, playerbbox, frame_id)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

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
    image_folder = (r"C:\hku\program\football_analysis\output1\output_gray")
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

        # Convert RGB colors to HSV
        # non_player_color1_hsv = cv2.cvtColor(np.uint8([[non_player_color1]]), cv2.COLOR_RGB2HSV)[0][0]
        # non_player_color2_hsv = cv2.cvtColor(np.uint8([[non_player_color2]]), cv2.COLOR_RGB2HSV)[0][0]
        # player_color_hsv = cv2.cvtColor(np.uint8([[player_color]]), cv2.COLOR_RGB2HSV)[0][0]
        # 输出颜色的 H 和 V 值
        # print(f"Player color - H: {player_color_hsv[0]}, V: {player_color_hsv[2]}")
        # print(f"Non-player color - H: {non_player_color1_hsv[0]}, V: {non_player_color1_hsv[2]}")
        # print(f"Non-player1 color - H: {non_player_color2_hsv[0]}, V: {non_player_color1_hsv[2]}")

        # 显示颜色的 RGB 值
        # print(f"Processing image: {os.path.basename(img_path)}")
        # print(f"Player color (RGB): {player_color}")
        # print(f"Non-player color (RGB): {non_player_color1}")
        # print(f"Non-player color2 (RGB): {non_player_color2}")
        # print(f"Non-player color3 (RGB): {non_player_color3}")
        # print(f"Non-player color4 (RGB): {non_player_color4}")
        # print(f"Non-player color5 (RGB): {non_player_color5}")

        # 创建纯色图像
        color_imgs = []
        for color in colors:
            img = np.zeros((300, 300, 3), np.uint8)
            img[:] = color
            color_imgs.append(img)
        # player_img = np.zeros((300, 300, 3), np.uint8)
        # player_img[:] = player_color
        #
        # non_player_img1 = np.zeros((300, 300, 3), np.uint8)
        # non_player_img1[:] = non_player_color1
        #
        # non_player_img2 = np.zeros((300, 300, 3), np.uint8)
        # non_player_img2[:] = non_player_color2

        # non_player_img3 = np.zeros((300, 300, 3), np.uint8)
        # non_player_img3[:] = non_player_color3

        # non_player_img4 = np.zeros((300, 300, 3), np.uint8)
        # non_player_img4[:] = non_player_color4
        #
        # non_player_img5 = np.zeros((300, 300, 3), np.uint8)
        # non_player_img5[:] = non_player_color5

        full_img = np.zeros((600, 900, 3), np.uint8)
        for i, color in enumerate(colors):
            full_img[0:300, i * 300:(i + 1) * 300] = color
            cv2.putText(full_img, f'H:{hsv_colors[i][0]} V:{hsv_colors[i][2]}', (i * 300 + 50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # full_img[0:300, 0:300] = player_color
        # full_img[0:300, 300:600] = non_player_color1
        # full_img[0:300, 600:900] = non_player_color2
        # full_img[300:600, 0:300] = non_player_color3
        # full_img[300:600, 300:600] = non_player_color4
        # full_img[300:600, 600:900] = non_player_color5

        # 在纯色图像上显示类别以及 H 和 V 值
        # cv2.putText(full_img, f'player H:{player_color_hsv[0]} V:{player_color_hsv[2]}', (50, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(full_img, f'Non-player1 H:{non_player_color1_hsv[0]} V:{non_player_color1_hsv[2]}', (350, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(full_img, f'Non-player2 H:{non_player_color2_hsv[0]} V:{non_player_color2_hsv[2]}', (650, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 在纯色图像上显示类别
        # cv2.putText(full_img, 'player', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(full_img, 'Non-player1', (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(full_img, 'Non-player2', (650, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
        # cv2.putText(full_img, 'Non-player3', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
        # cv2.putText(full_img, 'Non-player4', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
        # cv2.putText(full_img, 'Non-player5', (650, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
        # print("==================================")
        # print(color)
        cv2.imshow(os.path.basename(img_path), resized_img)
        # cv2.imshow('Non-player Color1', non_player_img1)
        # cv2.imshow('Non-player Color2', non_player_img2)
        # cv2.imshow('Non-player Color3', non_player_img3)
        # cv2.imshow('Non-player Color4', non_player_img4)
        # cv2.imshow('Non-player Color5', non_player_img5)
        # cv2.imshow('Player Color', player_img)
        cv2.imshow('full-image', full_img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyWindow(os.path.basename(img_path))

