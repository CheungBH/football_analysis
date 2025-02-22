import os

video_folder = r"D:\tmp\2.18"
# videos_path = [os.path.join(video_folder, video) for video in os.listdir(video_folder)]
output_folder = r"D:\tmp\2.18_output"
os.makedirs(output_folder, exist_ok=True)
model_path = "assets/checkpoints/best.pt"
court_img = r"D:\tmp\2.18\video_set_1474\court.jpg"

cmd_tpl = "python yolov9_bytetrack_pth_4_cropping.py --save_cropped_humans {} --video_path {} --output_dir {} --save_asset -m {} --court_image {} --show_video -o {}"

for video in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video)
    output_folder = os.path.join(output_folder, video.split('.')[0])
    output_dir = os.path.join(output_folder, "output")
    output_video = os.path.join(output_folder, "output_video.mp4")
    # output_topview = os.path.join(output_folder, "topview.jpg")
    save_cropped_path = os.path.join(output_folder, "save_cropped")
    cmd = cmd_tpl.format(save_cropped_path, video_path, output_dir, model_path, court_img, output_video)
    print(cmd)
    os.system(cmd)