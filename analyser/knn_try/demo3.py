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
            filtered_colors.append((closest_color, min_similarity))
            indices.append(index)

    return filtered_colors_raw, filtered_colors, indices


def process_images(folder_path, team_colors):
    dominant_colors = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
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
                cv2.putText(full_img, f'Index: {i} ', (i * 300 + 50, 500),  # 添加索引值
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(full_img, f'ratios:{round(ratios[i],2)}', (i * 300 + 50, 550),  # 添加索引值
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

            filtered_colors_raw, filtered_colors, indices = filter_colors(cluster_centers, team_colors)

            # 在第三行绘制过滤后的颜色块和相似度值
            for i in range(len(filtered_colors)):
                full_img[600:900, indices[i] * 300:(indices[i] + 1) * 300] = filtered_colors[i][0]  # 第三行
                cv2.putText(full_img, f'similarity:{str(int(filtered_colors[i][1]))}', (indices[i] * 300 + 50, 750),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(full_img, f'Index: {indices[i]}', (indices[i] * 300 + 50, 800),  # 添加索引值
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)


            cv2.imshow('full_img',full_img)
            cv2.imshow('raw',image)
            key = cv2.waitKey(0)
            second_color = np.array(filtered_colors_raw)
            initial_centers = np.array(filtered_colors)
            kmeans_2 = KMeans(n_clusters=len(initial_centers), init=initial_centers, n_init=1) # init  team_colors
            #kmeans_2.fit(second_color)



    return dominant_colors

# 获取A文件夹中的所有图片的第一个像素的颜色
team_colors = get_first_pixel_colors('/media/hkuit164/Backup/football_analysis/output/team_colors')
full_img = np.zeros((900, 1500, 3), np.uint8)
#hsv_team_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0].tolist() for color in team_colors]
for i, color in enumerate(team_colors):
        full_img[0:300, i * 300:(i + 1) * 300] = color  # 第一行
        cv2.putText(full_img, f'G:{team_colors[i][0]} B:{team_colors[i][1]} R:{team_colors[i][2]}', (i * 300 + 10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
dominant_colors = process_images('/media/hkuit164/Backup/football_analysis/output/111', team_colors)
# 打印结果
for i, color in enumerate(dominant_colors):
    print(f"图片 {i+1} 的主导颜色: {color}")
