import cv2
import numpy as np


def sliding_window_crop(image, window_size, crop_size):
    """
    Crop the bottom of an image using a sliding window approach.

    Parameters:
    - image: Input image (numpy array).
    - window_size: Size of the sliding window (tuple: (width, height)).
    - crop_size: Size of the crop (tuple: (crop_width, crop_height)).

    Returns:
    - List of cropped images (numpy arrays).
    """
    cropped_images = []
    f_id = []
    top_left = []
    img_height, img_width = image.shape[:2]

    # Define the starting point for the sliding window
    start_y = img_height - window_size[1]

    for i, x in enumerate(range(0, img_width - window_size[0] + 1, crop_size[0])):
        # Define the window's region of interest
        window = image[start_y:start_y + window_size[1], x:x + window_size[0]]

        # Append the cropped window to the list
        cropped_images.append(window)
        f_id.append(i)
        top_left.append((x, start_y))

    return {"images": cropped_images, "idx": f_id, "top_left": top_left}

# # Example usage:
# image = cv2.imread('../court_reference/soccer-field.png')
# cropped_images = sliding_window_crop(image, (100, 100), (50, 50))
# for i, cropped_image in enumerate(cropped_images):
#     cv2.imwrite(f'cropped_image_{i}.png', cropped_image)