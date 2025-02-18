import cv2
import numpy as np

# 定义鼠标点击事件的回调函数
def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = image[y, x].tolist()
        create_color_image(color)

def create_color_image(color):
    # 创建纯色图像
    color_image = np.zeros((300, 300, 3), np.uint8)
    color_image[:] = color

    # 显示纯色图像
    cv2.imshow('Color Image', color_image)
    cv2.imwrite('/media/hkuit164/Backup/football_analysis/datasets/game165/ref/goalkeeper.jpg', color_image)
    print(f"保存颜色图片: color_image.jpg")

# 读取图片
image = cv2.imread('/media/hkuit164/Backup/football_analysis/datasets/game165/check/player2/000063_127_482.jpg')  # 替换为你的图片文件名
cv2.imshow('Image', image)

# 设置鼠标回调函数
cv2.setMouseCallback('Image', get_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
