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

process_videos(r"E:\0220\videocut_output1\video_set_1100_reverse\output_video.mp4",
               r"E:\0220\videocut_output1\video_set_1100_reverse\top_view.mp4",
               'output_video.mp4')