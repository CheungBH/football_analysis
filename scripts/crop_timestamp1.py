
import os
from moviepy.editor import VideoFileClip

# Load the video file

# video_folder = ""
# video_paths = [
#     # "/Volumes/PortableSSD/futbol/0802/left_near/C0002.MP4",
#     # "/Volumes/PortableSSD/futbol/0802/left_far/C0024.MP4",
#     # "/Volumes/PortableSSD/futbol/0802/right_near/DSCF3627.MKV",
#     "/Volumes/PortableSSD/futbol/0802/right_far/C0467.MP4"
#     "/Volumes/PortableSSD/futbol/0802/right_far/C0467.MP4"
#     "/Volumes/PortableSSD/futbol/0802/right_far/C0467.MP4"
#     "/Volumes/PortableSSD/futbol/0802/right_far/C0467.MP4"
# ]
# # Define the end time relative to the start time in seconds
# clip_duration = 5

# Define a function to convert HH:MM:SS to seconds
def hms_to_seconds(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s

# Function to save the trimmed video
def save_trimmed_video(clip, output_path):
    clip.write_videofile(output_path, codec="libx264", fps=clip.fps)


def crop_clips(output_directory, begin_second, duration, video_folder):
    video_paths = [
        f"{video_folder}/output_video1.mp4",
        f"{video_folder}/output_video2.mp4",
        f"{video_folder}/output_video3.mp4",
        f"{video_folder}/output_video4.mp4"
    ]

    # Create a directory for the set of videos
    os.makedirs(output_directory, exist_ok=True)

    # Define the output paths
    output_paths = [
        f"{output_directory}/output_video1.mp4",
        f"{output_directory}/output_video2.mp4",
        f"{output_directory}/output_video3.mp4",
        f"{output_directory}/output_video4.mp4"
    ]

    # Trim and save each video
    for video_path, output_path in zip(video_paths, output_paths):
        end_seconds = begin_second + duration
        clip = VideoFileClip(video_path).subclip(begin_second, end_seconds)
        save_trimmed_video(clip, output_path)

    print(f"Trimmed videos saved in directory: {output_directory}")


if __name__ == '__main__':
    crop_clip_list = [
        ["/Volumes/ASSETS/tmp/2.18/video_set_161", "test", 10, 5]
    ]

    for crop_clip in crop_clip_list:
        crop_clips(crop_clip[1], crop_clip[2], crop_clip[3], crop_clip[0])