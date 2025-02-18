import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

input_folder = '/media/hkuit164/Backup/football_analysis/datasets/3/output_crop'
team_colors_folder = '/media/hkuit164/Backup/football_analysis/datasets/game1/ref'
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

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)
    image_2d = image.reshape(-1, 3)

    team_colors = get_first_pixel_colors(team_colors_folder)
    initial_centers1 = np.array(team_colors)
    kmeans1 = KMeans(n_clusters=len(initial_centers1), init=initial_centers1, n_init=1)
    kmeans1.fit(image_2d)
    cluster_centers = kmeans1.cluster_centers_ # 聚类颜色
    filtered_colors_raw, filtered_colors, indices, similarity = filter_colors(cluster_centers, team_colors,80) # caculate_distance,min_distance

    pixels=[]
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
    filtered_colors_raw2, filtered_colors2, indices2, similarity2 = filter_colors(color2, initial_centers2,99999)

    full_img = np.zeros((500, 700, 3), np.uint8)
    resized_img = cv2.resize(image, (500, 500))
    full_img[0:500, 0:500] = resized_img
    for i, color in enumerate(filtered_colors_raw2):
        full_img[i * 100:(i + 1) * 100, 500:600] = color
        full_img[i * 100:(i + 1) * 100, 600:700] = filtered_colors2[i]
        cv2.putText(full_img, 'similarity:{}'.format(similarity2[i]), (i * 300 + 50, 550),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    #cv2.imshow('Result', full_img)
    cv2.imwrite(f'output2/{image_name}.jpg',full_img)
    cv2.waitKey(1)

    cv2.destroyAllWindows()


