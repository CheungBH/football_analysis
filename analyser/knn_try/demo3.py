'''import os
import cv2
def process_file(filepath, ouput_txt_dir):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        id = data[0]
        x = float(data[1])
        y = float(data[2])
        w = float(data[3])
        h = float(data[4])
        y_ratio = crop_y/1080
        if 0 <= x < 1/3:
            x_new = x * 3
            y_new = (y - y_ratio) / (1 - y_ratio)
            w_new = w * 3
            h_new = h / (1 - y_ratio)
            output_file1 = os.path.join(ouput_txt_dir, os.path.splitext(os.path.basename(filepath))[0] + '_1.txt')
            if x_new + w_new > 1 or y_new - h_new/2 < 0:
                a=1
            else:
                with open(output_file1, 'a') as f_out:
                    f_out.write(f'{id} {x_new} {y_new} {w_new} {h_new}\n')
        if 1/6 <= x < 1/2:
            x_new = (x - 1/6) * 3
            y_new = (y - y_ratio) / (1 - y_ratio)
            w_new = w * 3
            h_new = h / (1 - y_ratio)
            output_file2 = os.path.join(ouput_txt_dir, os.path.splitext(os.path.basename(filepath))[0] + '_2.txt')
            if x_new + w_new/2 > 1 or y_new - h_new/2 < 0:
                a=1
            else:
                with open(output_file2, 'a') as f_out:
                    f_out.write(f'{id} {x_new} {y_new} {w_new} {h_new}\n')
        if 1/3 <= x < 2/3:
            x_new = (x - 1/3) * 3
            y_new = (y - y_ratio) / (1 - y_ratio)
            w_new = w * 3
            h_new = h / (1 - y_ratio)
            output_file3 = os.path.join(ouput_txt_dir, os.path.splitext(os.path.basename(filepath))[0] + '_3.txt')
            if x_new + w_new/2 > 1 or y_new - h_new/2 < 0:
                a=1
            else:
                with open(output_file3, 'a') as f_out:
                    f_out.write(f'{id} {x_new} {y_new} {w_new} {h_new}\n')
        if 1/2 <= x < 5/6:
            x_new = (x - 1/2) * 3
            y_new = (y - y_ratio) / (1 - y_ratio)
            w_new = w * 3
            h_new = h / (1 - y_ratio)
            output_file4 = os.path.join(ouput_txt_dir, os.path.splitext(os.path.basename(filepath))[0] + '_4.txt')
            if x_new + w_new/2 > 1 or y_new - h_new/2 < 0:
                a=1
            else:
                with open(output_file4, 'a') as f_out:
                    f_out.write(f'{id} {x_new} {y_new} {w_new} {h_new}\n')
        if 2/3 <= x <= 1:
            x_new = (x - 2/3) * 3
            y_new = (y - y_ratio) / (1 - y_ratio)
            w_new = w * 3
            h_new = h / (1 - y_ratio)
            output_file5 = os.path.join(ouput_txt_dir, os.path.splitext(os.path.basename(filepath))[0] + '_5.txt')
            if x_new + w_new/2 > 1 or y_new - h_new/2 < 0:
                a=1
            else:
                with open(output_file5, 'a') as f_out:
                    f_out.write(f'{id} {x_new} {y_new} {w_new} {h_new}\n')
        # if is_written1 ==True:


def process_image(filepath, ouput_img_dir):
    # 读取图片
    image = cv2.imread(filepath)
    # 调整大小到1920x1080
    resized_image = cv2.resize(image, (1920, 1080))
    # 保留y值在440到1080的部分
    cropped_image = resized_image[crop_y:1080, :]

    height, width, _ = cropped_image.shape
    segment_width = width // 6

    # 分割并保存图像
    for i in range(5):
        if i == 0:
            segment = cropped_image[:, :segment_width*2]
            output_file = os.path.join(ouput_img_dir, os.path.splitext(os.path.basename(filepath))[0] + '_1.jpg')
        elif i == 1:
            segment = cropped_image[:, segment_width:3*segment_width]
            output_file = os.path.join(ouput_img_dir, os.path.splitext(os.path.basename(filepath))[0] + '_2.jpg')
        elif i == 2:
            segment = cropped_image[:, 2*segment_width:4*segment_width]
            output_file = os.path.join(ouput_img_dir, os.path.splitext(os.path.basename(filepath))[0] + '_3.jpg')
        elif i == 3:
            segment = cropped_image[:, 3*segment_width:5*segment_width]
            output_file = os.path.join(ouput_img_dir, os.path.splitext(os.path.basename(filepath))[0] + '_4.jpg')
        elif i == 4:
            segment = cropped_image[:, 4*segment_width:]
            output_file = os.path.join(ouput_img_dir, os.path.splitext(os.path.basename(filepath))[0] + '_5.jpg')

        cv2.imwrite(output_file, segment)

def process_directory(input_dir, ouput_dir,cnt):
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            process_file(filepath, ouput_dir)
        elif filename.endswith('.jpg'):
            filepath = os.path.join(input_dir, filename)
            process_image(filepath, ouput_dir)
        cnt+=1
        print(cnt)


if __name__ == "__main__":
    input_txt_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/labels/val"
    ouput_txt_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/labels1/val"
    input_img_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/images/val"
    ouput_img_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/images1/val"
    crop_y = 440
    cnt=0
    os.makedirs(ouput_img_dir, exist_ok=True)
    process_directory(input_img_dir, ouput_img_dir,cnt)
    os.makedirs(ouput_txt_dir, exist_ok=True)
    process_directory(input_txt_dir, ouput_txt_dir,cnt)
'''



import cv2
import os

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    x_center, y_center, width, height = bbox
    h, w, _ = image.shape
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

def process_files(img_dir, txt_dir):
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            txt_filepath = os.path.join(txt_dir, filename)
            img_filename = os.path.splitext(filename)[0] + '.jpg'
            img_filepath = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_filepath):
                continue

            image = cv2.imread(img_filepath)

            with open(txt_filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                x = float(data[1])
                y = float(data[2])
                w = float(data[3])
                h = float(data[4])
                bbox = (x, y, w, h)
                image = draw_bounding_box(image, bbox)
            resize_img = cv2.resize(image,(1080,720))
            cv2.imshow('Visualized Image', resize_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    txt_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/labels1/train"

    img_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/images1/train"

    process_files(img_dir, txt_dir)


'''import os

def compare_and_delete(txt_dir, jpg_dir):
    # 获取txt文件的基本文件名（不带扩展名）
    txt_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(txt_dir) if filename.endswith('.txt')}

    # 遍历jpg文件夹
    for filename in os.listdir(jpg_dir):
        if filename.endswith('.jpg'):
            # 获取jpg文件的基本文件名
            jpg_basename = os.path.splitext(filename)[0]
            # 检查jpg文件是否在txt文件列表中
            if jpg_basename not in txt_filenames:
                # 如果不在，则删除
                os.remove(os.path.join(jpg_dir, filename))
                print(f"Deleted: {filename}")

if __name__ == "__main__":
    txt_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/labels1/train"
    jpg_dir = "/media/hkuit164/Backup/yolov9/datasets/all_YOLO_new/images1/train"
    compare_and_delete(txt_dir, jpg_dir)
'''
