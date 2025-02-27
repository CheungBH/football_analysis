import os

cmds = [
    "python yolov9_bytetrack_pth_4_cropping.py --save_cropped_humans E:/0220/videocut_output1/video_set_855_2min\save_cropped --video_path E:/0220/videocut/video_set_855_2min --use_json --output_dir E:/0220/videocut_output1/video_set_855_2min/output --save_asset -m assets/checkpoints/best.pt --court_image D:/tmp/2.18/video_set_1474\court.jpg -o E:/0220/videocut_output1/video_set_855_2min/output_video.mp4 --show_video --stop_at -1",
    "python yolov9_bytetrack_pth_4_cropping.py --save_cropped_humans E:/0220/videocut_output1/video_set_1248/save_cropped --video_path D:/tmp/2.18/video_set_1248 --use_json --output_dir E:/0220/videocut_output1/video_set_1248/output --save_asset -m assets/checkpoints/best.pt --court_image D:/tmp/2.18/video_set_1474/court.jpg -o E:/0220/videocut_output1/video_set_1248/output_video.mp4 --show_video --stop_at -1",
]

for cmd in cmds:
    os.system(cmd)