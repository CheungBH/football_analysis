import cv2
from moviepy.editor import VideoFileClip

def resample_videos(video_paths, target_fps):
    for path in video_paths:
        # 读取视频
        cap = cv2.VideoCapture(path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / original_fps)

        # 使用 MoviePy 调整视频帧率
        clip = VideoFileClip(path)
        clip_resampled = clip.set_fps(target_fps)

        # 保存调整后的视频
        output_path = f"resampled_{path}"
        clip_resampled.write_videofile(output_path, codec="libx264", fps=target_fps)
        print(f"Video {path} resampled from {original_fps} FPS to {target_fps} FPS and saved as {output_path}")



def select_start_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (1080, 720))
            cv2.imshow(f'Select start frame for {video_path}', resized_frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 按下空格键选择当前帧作为开始帧
                print(f'Start frame for {video_path} selected: {frame_id}')
                break
            elif key == 27:  # 按下ESC键退出
                break

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        return frame_id





