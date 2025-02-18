
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from shutil import copyfile

def get_first_pixel_colors(folder_path):
    team_color = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, file_name))
            first_pixel_color = image[0, 0]
            team_color.append(first_pixel_color.tolist())
    return team_color

def get_dominant_colors(image, initial_centers):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=1)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

def color_difference(color1, color2):
    return np.linalg.norm(color1 - color2)

def process_images(input_folder, output_folder, threshold=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        team_colors = get_first_pixel_colors('/media/hkuit164/Backup/football_analysis/datasets/game50/ref')
        initial_centers = np.array(team_colors)
        dominant_colors = get_dominant_colors(image, initial_centers)
        for color in dominant_colors:
            for prev_color in team_colors:
                if color_difference(color, prev_color) < threshold:
                    save_folder = os.path.join(output_folder, str(prev_color))
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    copyfile(image_path, os.path.join(save_folder, image_name))
                    break

# 示例调用
input_folder = '/media/hkuit164/Backup/sn-tracking/data/tracking/output/50'
output_folder = '/media/hkuit164/Backup/football_analysis/datasets/game50/test' # 示例颜色
process_images(input_folder, output_folder)
