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

# import cv2
# # 打开视频文件
# cap = cv2.VideoCapture('/media/hkuit164/Backup/football_analysis/result1.mp4')
#
# # 检查是否成功打开视频文件
# if not cap.isOpened():
#     print("Error: Cannot open video file.")
#     exit()
#
# # 获取视频属性
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# # 创建视频写入对象
# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# # 初始化文本信息
# lowspeed = False
# color = (0, 255, 0)  # 初始化颜色为绿色
#
# # 打开txt文件以追加写入模式
# with open('output.txt', 'a') as txt_file:
#     frame_id = 0
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#         # 显示当前帧
#         cv2.imshow('Frame', frame)
#
#         # 获取按键输入
#         key = cv2.waitKey(0) & 0xFF
#
#         if key == ord('1'):
#             lowspeed = False
#             color = (0, 255, 0)  # 绿色
#         elif key == ord('2'):
#             lowspeed = True
#             color = (0, 0, 255)  # 红色
#         elif key == ord('3'):
#             frame_id += 1
#             continue  # 跳到下一帧
#         elif key == 27:  # 按下 ESC 键退出
#             break
#
#         # 设置文本内容和位置
#         text = f"lowspeed: {lowspeed}"
#         position = (10, 30)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#
#         # 在帧上放置文本
#         cv2.putText(frame, text, position, font, 1, color, 2, cv2.LINE_AA)
#
#         # 将处理后的帧写入新视频
#         out.write(frame)
#
#         # 打印并保存frame_id和text
#         print(f"Frame ID: {frame_id}, Text: {text}")
#         txt_file.write(f"[{frame_id}, {text}]\n")
#       # 增加frame_id
#         frame_id += 1
#
# # 释放视频对象和写入对象，并关闭所有窗口
# cap.release()
# out.release()
# cv2.destroyAllWindows()



import cv2

def read_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            frame_id, text = line.strip()[1:-1].split(", ")
            frame_id = int(frame_id)
            text = text.split(": ")[1] == "True"
            data.append((frame_id, text))
    return data

# 读取txt文件内容
txt_file_path = 'output.txt'
data = read_txt(txt_file_path)

# 打开视频文件
input_video = 'output_video.mp4'
cap = cv2.VideoCapture(input_video)

# 检查是否成功打开视频文件
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 创建视频写入对象
out = cv2.VideoWriter('output_video1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 初始化帧索引
frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 在对应帧上放置文本
    for item in data:
        if item[0] == frame_id:
            color = (0, 0, 255) if item[1] else (0, 255, 0)
            text = f"lowspeed: {item[1]}"
            position = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # 在帧上放置文本
            cv2.putText(frame, text, position, font, 1, color, 2, cv2.LINE_AA)

    # 将处理后的帧写入新视频
    out.write(frame)

    # 增加帧索引
    frame_id += 1

# 释放视频对象和写入对象，并关闭所有窗口
cap.release()
out.release()
cv2.destroyAllWindows()
