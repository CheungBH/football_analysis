import os
import shutil

video_folder = r"D:\tmp\3.1\demo_videos"

output_folder_root = r"D:\tmp\3.2\demo_videos_json"
os.makedirs(output_folder_root, exist_ok=True)

video_paths = [os.path.join(video_folder, video) for video in os.listdir(video_folder)]

for video_path in video_paths:
    video_name = os.path.basename(video_path).split('.')[0]
    # output_folder = os.path.join(output_folder_root, video_name)
    # os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(video_folder, video_name, "yolo.json")
    output_path = os.path.join(output_folder_root, "{}.json".format(video_name))
    shutil.copy(json_path, output_path)

