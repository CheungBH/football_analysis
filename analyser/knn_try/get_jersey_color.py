import cv2
import numpy as np
import json
import os


def get_jersey_color(image_path, white_threshold=0.8):
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Step 4: Calculate the proportion of white pixels
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    white_proportion = white_pixels / total_pixels

    # Step 5: Classify as white if the proportion of white pixels is above the threshold
    if white_proportion > white_threshold:
        return (255, 255, 255)

    # Step 6: Find contours of non-white regions
    _, thresh_inv = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Create a mask for the non-white regions
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Step 8: Compute the dominant color of the non-white regions
    mask = cv2.bitwise_and(image, image, mask=mask)
    non_white_pixels = mask[np.where((mask != [0, 0, 0]).all(axis=2))]

    if len(non_white_pixels) == 0:
        return "White (No significant non-white regions detected)"

    avg_color = np.mean(non_white_pixels, axis=0)

    return avg_color.astype(int).tolist()


def process_images(input_folder, output_folder, output_json):
    image_data = {}
    roles = ["player1", "player2", "goalkeeper1", "goalkeeper2", "referee"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, role in enumerate(roles):
        # file_nam
        file_path = os.path.join(input_folder, role + ".jpg")

        jersey_color = get_jersey_color(file_path)

        # Save the BGR color data
        image_data[idx] = jersey_color

        # Create the pure color image if the jersey color is not classified as "White"
        if isinstance(jersey_color, list):
            color_image = np.full((100, 100, 3), jersey_color, dtype=np.uint8)
            color_image_path = os.path.join(output_folder, f"{role}")
            # cv2.imwrite(color_image_path, color_image)
            print(f"Processed and saved pure color image for: {role}")
        else:
            print(f"Image {role} classified as White or no significant non-white regions detected.")

    # Save the JSON data
    with open(output_json, 'w') as json_file:
        json.dump(image_data, json_file, indent=4)
    print(f"Saved BGR color data to {output_json}")


if __name__ == '__main__':
    # Example usage
    input_folder = 'knn_assets/jersey'
    output_folder = 'knn_assets/jersey_output'
    output_json = 'knn_assets/jersey_output/color.json'
    process_images(input_folder, output_folder, output_json)