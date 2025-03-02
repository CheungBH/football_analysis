import cv2

# Get the frame number of a video

video = cv2.VideoCapture(r"D:\tmp\3.1\demo_videos\commit_foul\output_video4.mp4")
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)