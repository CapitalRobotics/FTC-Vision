import os
import xml.etree.ElementTree as ET

def update_xml_paths(annotation_dir, image_base_dir):
    """
    Updates the folder and path elements in XML files to include class subdirectories.

    Args:
        annotation_dir (str): Path to the directory containing XML annotations.
        image_base_dir (str): Path to the base directory of the images (e.g., dataset/Images/train or val).
    """
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get class name from <object>/<name>
        class_name = root.find('object/name').text.lower()  # Ensure lowercase
        if not class_name:
            print(f"Skipping {xml_file}: No class name found.")
            continue

        # Update <folder>
        folder_element = root.find('folder')
        if folder_element is not None:
            folder_element.text = os.path.join(image_base_dir, class_name)

        # Update <path>
        filename = root.find('filename').text
        path_element = root.find('path')
        if path_element is not None:
            path_element.text = os.path.join(image_base_dir, class_name, filename)

        # Save updated XML
        tree.write(xml_path)
        print(f"Updated {xml_file} with new paths.")

# Define paths
train_annotation_dir = "dataset/Annotations/train"
val_annotation_dir = "dataset/Annotations/val"
train_image_base_dir = "dataset/Images/train"
val_image_base_dir = "dataset/Images/val"

# Update XML paths for train and val
update_xml_paths(train_annotation_dir, train_image_base_dir)
update_xml_paths(val_annotation_dir, val_image_base_dir)