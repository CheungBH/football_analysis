import os
from PIL import Image

def crop_images_in_folder(input_folder, output_folder, crop_size=(480, 480), step_size=240):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            width, height = image.size

            if height < crop_size[1]:
                print(f"Skipping {filename}, height is less than crop height.")
                continue

            y = height - crop_size[1] - 50
            for x in range(0, width - crop_size[0] + 1, step_size):
                cropped_image = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
                cropped_filename = f"{os.path.splitext(filename)[0]}_cropped_{x}_{y}.png"
                cropped_image.save(os.path.join(output_folder, cropped_filename))
                print(f"Saved cropped image: {cropped_filename}")

            # Check if the last window needs to be processed
            if (width - crop_size[0]) % step_size != 0:
                x = width - crop_size[0]
                cropped_image = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
                cropped_filename = f"{os.path.splitext(filename)[0]}_cropped_{x}_{y}.png"
                cropped_image.save(os.path.join(output_folder, cropped_filename))
                print(f"Saved cropped image: {cropped_filename}")


if __name__ == "__main__":
    input_folder = '/Users/cheungbh/Downloads/soccer_field_200_dataset'  # Change to your input folder path
    output_folder = '/Users/cheungbh/Downloads/soccer_field_200_dataset_out'  # Change to your output folder path
    crop_images_in_folder(input_folder, output_folder)