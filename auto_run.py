import os

video_folder = r"D:\tmp\3.1\demo_videos"
# videos_path = [os.path.join(video_folder, video) for video in os.listdir(video_folder)]
output_folder_root = r"D:\tmp\3.5\demo_videos_output"
os.makedirs(output_folder_root, exist_ok=True)
model_path = r"D:\tmp\3.5\best.pt"
court_img = r"E:\0220\videocut\court.jpg"

use_save_box = False
use_save_team = False
stop_at = -1 # -1 means no limit

cmd_tpl = ("python yolov9_bytetrack_pth_4_cropping.py --save_cropped_humans {} --video_path {} --use_json "
           "--output_dir {} --save_asset -m {} --court_image {} --show_video -o {} --stop_at {}")
if use_save_box:
    cmd_tpl += " --use_saved_box"
if use_save_team:
    cmd_tpl += " --use_saved_team"


for video in os.listdir(video_folder):
    if not os.path.isdir(os.path.join(video_folder, video)):
        continue
    video_path = os.path.join(video_folder, video)
    output_folder = os.path.join(output_folder_root, video.split('.')[0])
    output_dir = os.path.join(output_folder, "output")
    output_video = os.path.join(output_folder, "output_video.mp4")
    # output_topview = os.path.join(output_folder, "topview.jpg")
    save_cropped_path = os.path.join(output_folder, "save_cropped")
    cmd = cmd_tpl.format(save_cropped_path, video_path, output_dir, model_path, court_img, output_video, stop_at)
    print(cmd)
    os.system(cmd)