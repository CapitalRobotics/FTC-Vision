import os
import shutil
import xml.etree.ElementTree as ET

def sort_images_by_class(image_dir, annotation_dir):
    """
    Sort images into subdirectories (red, blue, yellow) based on XML annotations.

    Args:
        image_dir (str): Path to the directory containing images.
        annotation_dir (str): Path to the directory containing corresponding XML files.
    """
    # Ensure subdirectories exist for each class
    classes = ['red', 'blue', 'yellow']
    for class_name in classes:
        os.makedirs(os.path.join(image_dir, class_name), exist_ok=True)

    # Iterate over all XML files
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith('.xml'):
            continue

        # Parse the XML file
        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract the image filename and class name
        filename = root.find('filename').text
        class_name = root.find('object/name').text.lower()  # Ensure lowercase class names

        if class_name not in classes:
            print(f"Skipping unknown class '{class_name}' in {xml_file}.")
            continue

        # Move the image to the appropriate subdirectory
        src_image_path = os.path.join(image_dir, filename)
        dst_image_path = os.path.join(image_dir, class_name, filename)

        if os.path.exists(src_image_path):
            shutil.move(src_image_path, dst_image_path)
            print(f"Moved {src_image_path} -> {dst_image_path}")
        else:
            print(f"Image {src_image_path} not found, skipping.")

# Define paths
train_image_dir = "dataset/Images/train"
train_annotation_dir = "dataset/Annotations/train"

val_image_dir = "dataset/Images/val"
val_annotation_dir = "dataset/Annotations/val"

# Sort images for train and val datasets
sort_images_by_class(train_image_dir, train_annotation_dir)
sort_images_by_class(val_image_dir, val_annotation_dir)
