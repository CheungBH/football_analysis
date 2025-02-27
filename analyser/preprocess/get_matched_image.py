import os
import shutil

def move_matching_images(folder1, folder2, dest_folder):
    for filename in os.listdir(folder1):
        if os.path.isfile(os.path.join(folder1, filename)):
            if os.path.exists(os.path.join(folder2, filename)):
                shutil.move(os.path.join(folder1, filename), os.path.join(dest_folder, filename))
                print(f"Moved {filename} to {dest_folder}")

# Replace these paths with your folder paths
folder1 = "path/to/first/folder"
folder2 = "path/to/second/folder"
dest_folder = "path/to/third/folder"

move_matching_images(folder1, folder2, dest_folder)