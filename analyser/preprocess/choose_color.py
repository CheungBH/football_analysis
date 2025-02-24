import cv2
import numpy as np
import os

# Global variables to hold current image and its name
current_image = None
current_image_name = None
output_folder = '/Users/cheungbh/Documents/lab_code/football_analysis/analyser/knn_try/knn_assets/jersey_click'  # Change to your output folder path


# Define the mouse click event callback function
def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = current_image[y, x].tolist()
        create_color_image(color, current_image_name)


def create_color_image(color, image_name):
    # Create pure color image
    color_image = np.zeros((300, 300, 3), np.uint8)
    color_image[:] = color

    # Display pure color image
    cv2.imshow('Color Image', color_image)

    # Save the color image with a unique name
    basename = os.path.splitext(os.path.basename(image_name))[0]
    save_path = os.path.join(output_folder, f'{basename}_color.jpg')
    cv2.imwrite(save_path, color_image)
    print(f"Saved color image: {save_path}")


def process_images_in_folder(input_folder):
    global current_image, current_image_name

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            current_image_name = image_path
            current_image = cv2.imread(image_path)

            # Display the image
            cv2.imshow('Image', current_image)

            # Set mouse callback function
            cv2.setMouseCallback('Image', get_color)

            # Wait until a key is pressed
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    input_folder = '/Users/cheungbh/Documents/lab_code/football_analysis/analyser/knn_try/knn_assets/jersey'  # Change to your input folder path
    process_images_in_folder(input_folder)
