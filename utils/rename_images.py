import os

# Paths to train and val image directories
TRAIN_IMAGES_DIR = "dataset/images/train"
VAL_IMAGES_DIR = "dataset/images/val"

def rename_image_files(images_dir):
    # List all files in the directory
    files = os.listdir(images_dir)

    for file in files:
        # Check if the file starts with "IMG_" and ends with ".JPG"
        if file.startswith("IMG_") and file.endswith(".JPG"):
            old_path = os.path.join(images_dir, file)
            # Rename file to lowercase (e.g., IMG_5135.JPG -> img_5135.jpg)
            new_filename = file.lower()
            new_path = os.path.join(images_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
        elif file.startswith("IMG_") and file.endswith(".jpg"):
            old_path = os.path.join(images_dir, file)
            # Convert "IMG_" to "img_"
            new_filename = file.replace("IMG_", "img_")
            new_path = os.path.join(images_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    # Rename files in the train directory
    print("Renaming files in train directory...")
    rename_image_files(TRAIN_IMAGES_DIR)

    # Rename files in the val directory
    print("\nRenaming files in val directory...")
    rename_image_files(VAL_IMAGES_DIR)

    print("\nImage file renaming completed!")
    