import os

TRAIN_IMAGES_DIR = "dataset/images/train"
VAL_IMAGES_DIR = "dataset/images/val"

def rename_image_files(images_dir):
    files = os.listdir(images_dir)

    for file in files:
        if file.startswith("IMG_") and file.endswith(".JPG"):
            old_path = os.path.join(images_dir, file)
            new_filename = file.lower()
            new_path = os.path.join(images_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
        elif file.startswith("IMG_") and file.endswith(".jpg"):
            old_path = os.path.join(images_dir, file)
            new_filename = file.replace("IMG_", "img_")
            new_path = os.path.join(images_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    print("Renaming files in train directory...")
    rename_image_files(TRAIN_IMAGES_DIR)

    print("\nRenaming files in val directory...")
    rename_image_files(VAL_IMAGES_DIR)

    print("\nImage file renaming completed!")
    
