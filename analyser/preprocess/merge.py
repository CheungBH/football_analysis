import cv2

def process_videos(video1_path, video2_path, output_path):
    # Open the first video
    cap1 = cv2.VideoCapture(video1_path)
    # Open the second video
    cap2 = cv2.VideoCapture(video2_path)

    # Get video properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height * 2))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Process the first video frame
        top_half = frame1[0:height//2, :]
        bottom_half = frame1[height//2:height, :]

        # Swap the left/right sides of the bottom half
        bottom_left = bottom_half[:, 0:width//2]
        bottom_right = bottom_half[:, width//2:width]
        swapped_bottom_half = cv2.hconcat([bottom_right, bottom_left])

        # Combine the swapped bottom half with the top half
        processed_frame1 = cv2.vconcat([top_half, swapped_bottom_half])

        # Resize the second frame to match the width of the first
        frame2_resized = cv2.resize(frame2, (width, height))

        # Merge the processed frame with the resized second frame vertically
        merged_frame = cv2.vconcat([processed_frame1, frame2_resized])

        # Write the merged frame to the output video
        out.write(merged_frame)

    # Release everything
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage

if __name__ == '__main__':
    import os, shutil
    src_folder = r"D:\tmp\3.10\demo_videos_output"
    out_folder = r"output_merged"
    raw_folder = r"output_raw"
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    for folder in os.listdir(src_folder):
        video1_path = os.path.join(src_folder, folder, "output_video.mp4")
        video2_path = os.path.join(src_folder, folder, "top_view.mp4")
        output_path = os.path.join(out_folder, f"{folder}_merged.mp4")
        if os.path.exists(output_path):
            continue

        process_videos(video1_path, video2_path, output_path)
        raw_output_path = os.path.join(raw_folder, f"{folder}_raw.mp4")
        shutil.copy(video1_path, raw_output_path)
        raw_tv_output_path = os.path.join(raw_folder, f"{folder}_top_view.mp4")
        shutil.copy(video2_path, raw_tv_output_path)

# process_videos(r"D:\tmp\2.24_output1\video_set_1486\output_video.mp4",
#                r"D:\tmp\2.24_output1\video_set_1486\top_view.mp4",
#                'output_video.mp4')