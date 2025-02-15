import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def get_first_pixel_colors(folder_path):
    team_color = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, file_name))
            first_pixel_color = image[0, 0]
            team_color.append(first_pixel_color.tolist())
    return team_color

def filter_colors(clustered_colors, team_colors, threshold=80):
    filtered_colors_raw = []
    filtered_colors = []
    indices = []
    similarities = []

    for index, color1 in enumerate(clustered_colors):
        min_similarity = float('inf')
        closest_color = None

        for color2 in team_colors:
            similarity = np.linalg.norm(np.array(color1) - np.array(color2))
            if similarity < min_similarity:
                min_similarity = similarity
                closest_color = color2

        if min_similarity < threshold:
            filtered_colors_raw.append([int(c) for c in color1.tolist()])
            filtered_colors.append(closest_color)
            similarities.append(min_similarity)
            indices.append(index)

    return filtered_colors_raw, filtered_colors, indices, similarities


def process_images(folder_path, team_colors):
    dominant_colors = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            full_img = np.zeros((900, 1500, 3), np.uint8)
            # hsv_team_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0].tolist() for color in team_colors]
            for i, color in enumerate(team_colors):
                full_img[0:300, i * 300:(i + 1) * 300] = color  # 第一行
                cv2.putText(full_img, f'G:{team_colors[i][0]} B:{team_colors[i][1]} R:{team_colors[i][2]}',
                            (i * 300 + 10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
            image = cv2.imread(os.path.join(folder_path, file_name))
            image_2d = image.reshape(-1, 3)

            # KMeans聚类出6种颜色
            initial_centers = np.array(team_colors)
            kmeans = KMeans(n_clusters=len(team_colors), init=initial_centers, n_init=1) # init  team_colors
            kmeans.fit(image_2d)
            labels=kmeans.labels_
            label_counts = np.bincount(labels)
# 计算每个类的占有比例
            total_samples = len(labels)
            ratios = label_counts / total_samples
            cluster_centers = kmeans.cluster_centers_
            team_clustered_colors = cluster_centers.tolist()
            #hsv_clustered_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0].tolist() for color in team_clustered_colors]
            for i, color in enumerate(team_clustered_colors):
                full_img[300:600, i * 300:(i + 1) * 300] = color  # 第二行
                cv2.putText(full_img, f'G:{int(team_clustered_colors[i][0])} B:{int(team_clustered_colors[i][1])} R:{int(team_clustered_colors[i][2])}',
                            (i * 300 + 10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(full_img, f'Index: {i} ', (i * 300 + 50, 500),  # 添加索引值
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(full_img, f'ratios:{round(ratios[i],2)}', (i * 300 + 50, 550),  # 添加索引值
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

            filtered_colors_raw, filtered_colors, indices, similarity = filter_colors(cluster_centers, team_colors)

            # 在第三行绘制过滤后的颜色块和相似度值
            for i in range(len(filtered_colors_raw)):
                full_img[600:900, indices[i] * 300:(indices[i] + 1) * 300] = filtered_colors[i]  # 第三行
                cv2.putText(full_img, f'G:{int(filtered_colors[i][0])} B:{int(filtered_colors[i][1])} R:{int(filtered_colors[i][2])}',
                            (indices[i] * 300 + 10, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(full_img, 'similarity:{}'.format(similarity[i]), (indices[i] * 300 + 50, 750),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)


            # key = cv2.waitKey(0)
            # Keep the unique colors from a list of colors
            filtered_team_color = list(set([tuple(color) for color in filtered_colors]))

            second_color = np.array(filtered_team_color)
            # initial_centers = np.array(filtered_colors)
            kmeans_2 = KMeans(n_clusters=len(second_color), init=second_color, n_init=1) # init  team_colors
            kmeans_2.fit(second_color)
            labels_2 = kmeans_2.predict(image_2d)
            # labels_2 = kmeans_2.labels_
            label_counts_2 = np.bincount(labels_2)
            # label_counts = np.bincount(labels)
            # 计算每个类的占有比例
            total_samples = len(labels_2)
            ratios_2 = label_counts_2 / total_samples
            team_clustered_colors2 = kmeans_2.cluster_centers_
            new_img = np.zeros((600, 300*len(team_clustered_colors2), 3), np.uint8)

            filtered_colors_raw_2, filtered_colors_2, indices_2, similarity_2 = filter_colors(team_clustered_colors2, filtered_colors, threshold=1000)

            for i, color in enumerate(filtered_team_color):
                new_img[0:300, i * 300:(i + 1) * 300] = color  # 第一行
                cv2.putText(new_img, f'G:{filtered_team_color[i][0]} B:{filtered_team_color[i][1]} R:{filtered_team_color[i][2]}',
                            (i * 300 + 10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

            for i, color in enumerate(filtered_colors_raw_2):
                new_img[300:600, i * 300:(i + 1) * 300] = color  # 第二行
                cv2.putText(new_img, f'G:{int(filtered_colors_raw_2[i][0])} B:{int(filtered_colors_raw_2[i][1])} R:{int(filtered_colors_raw_2[i][2])}',
                            (i * 300 + 10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(new_img, f'Index: {i} ', (i * 300 + 50, 500),  # 添加索引值
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(new_img, f'ratios:{round(ratios_2[i],2)}', (i * 300 + 50, 550),  # 添加索引值
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('cluster_second',new_img)
            cv2.imshow('full_img', full_img)
            cv2.imshow('raw', image)
            cv2.waitKey(0)



team_colors = get_first_pixel_colors('knn_assets/team_colors')

process_images('knn_assets/team_colors/raw_img/yellow', team_colors)

