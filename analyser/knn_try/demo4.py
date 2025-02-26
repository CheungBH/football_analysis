#
# import os
# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# from shutil import copyfile
#
# def get_first_pixel_colors(folder_path):
#     team_color = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.jpg') or file_name.endswith('.png'):
#             image = cv2.imread(os.path.join(folder_path, file_name))
#             first_pixel_color = image[0, 0]
#             team_color.append(first_pixel_color.tolist())
#     return team_color
#
# def get_dominant_colors(image, initial_centers):
#     pixels = image.reshape((-1, 3))
#     kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=1)
#     kmeans.fit(pixels)
#     return kmeans.cluster_centers_
#
# def color_difference(color1, color2):
#     return np.linalg.norm(color1 - color2)
#
# def process_images(input_folder, output_folder, threshold=30):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for image_name in os.listdir(input_folder):
#         image_path = os.path.join(input_folder, image_name)
#         image = cv2.imread(image_path)
#         if image is None:
#             continue
#         team_colors = get_first_pixel_colors('/media/hkuit164/Backup/football_analysis/datasets/game50/ref')
#         initial_centers = np.array(team_colors)
#         dominant_colors = get_dominant_colors(image, initial_centers)
#         for color in dominant_colors:
#             for prev_color in team_colors:
#                 if color_difference(color, prev_color) < threshold:
#                     save_folder = os.path.join(output_folder, str(prev_color))
#                     if not os.path.exists(save_folder):
#                         os.makedirs(save_folder)
#                     copyfile(image_path, os.path.join(save_folder, image_name))
#                     break
#
# # 示例调用
# input_folder = '/media/hkuit164/Backup/sn-tracking/data/tracking/output/50'
# output_folder = '/media/hkuit164/Backup/football_analysis/datasets/game50/test' # 示例颜色
# process_images(input_folder, output_folder)
#
#

import math

def calculate_speeds(coordinates):
    # 过滤掉包含 [-1, -1] 之间的点
    valid_coordinates = []
    skip_next = False
    for i in range(len(coordinates)):
        if coordinates[i] == [-1, -1]:
            skip_next = True
        elif skip_next:
            skip_next = False
        else:
            valid_coordinates.append(coordinates[i])

    # 计算速度
    speeds = []
    for i in range(1, len(valid_coordinates)):
        x1, y1 = valid_coordinates[i-1]
        x2, y2 = valid_coordinates[i]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = distance  # 如果你有时间间隔，可以除以时间得到速度
        speeds.append(speed)

    return speeds

coordinates1 = [[928.0203195233446, 299.19832258583824], [928.0444568007517, 299.5180064589547], [928.1738855150277, 300.71975708709306], [928.0736959228426, 301.2244653501388], [928.1370205526339, 302.42930408797145], [927.6972478853295, 302.27897116494853], [927.7355617850061, 303.84453094046705], [927.3834932473244, 302.87673402210345],
                [927.5324325324189, 304.1012701441836],
                [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
                [927.9897165565454, 308.55162348153556]]
valid_spped=[]
valid_distance=[]
for i in range(len(coordinates1)-1):
    if coordinates1[i] !=[-1,-1] and coordinates1[i+1]!=[-1,-1]:
        x1, y1 = coordinates1[i]
        x2, y2 = coordinates1[i+1]
        valid_spped.append([coordinates1[i],coordinates1[i+1]])
        valid_distance.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))




coordinates2 = [[927.9897165565454, 308.55162348153556], [928.2502951918485, 308.9937624888397], [928.4401619636683, 309.0279664704596], [927.8804017463923, 306.51647806886353], [927.4174124926274, 304.6826927906266], [927.0136459114034, 302.9035711090442], [926.8486690688593, 302.24062467695836],
                [926.7420035232525, 302.0049505495917], [926.6734644311285, 301.9284146801014]]

speeds1 = calculate_speeds(coordinates1)
speeds2 = calculate_speeds(coordinates2)

print("Speeds for first segment:", speeds1)
print("Speeds for second segment:", speeds2)
