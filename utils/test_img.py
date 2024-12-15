import os
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

def draw_bounding_box(image, box, label):
    """
    Draws a bounding box and label on the image.

    Args:
        image (numpy.ndarray): The image to draw on.
        box (tuple): The bounding box as (xmin, ymin, xmax, ymax).
        label (str): The label of the object.
    """
    xmin, ymin, xmax, ymax = box
    color = (0, 255, 0)  # Green box
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Draw the bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

    # Put the label
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = xmin
    text_y = ymin - 10 if ymin - 10 > 10 else ymin + 10
    cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

def test_random_xml(annotation_dir, image_base_dir):
    """
    Randomly selects an XML file, opens the corresponding image, and displays it with annotations.

    Args:
        annotation_dir (str): Path to the directory containing XML files.
        image_base_dir (str): Path to the base directory of the images.
    """
    # Get a list of XML files
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]

    if not xml_files:
        print("No XML files found in the specified directory.")
        return

    # Randomly select an XML file
    selected_xml = random.choice(xml_files)
    xml_path = os.path.join(annotation_dir, selected_xml)

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image details
    filename = root.find('filename').text
    folder = root.find('folder').text
    label = root.find('object/name').text  # Use the class name to determine the subdirectory
    # Build the full path to the image
    image_path = os.path.join(image_base_dir, label, filename)

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        draw_bounding_box(image, (xmin, ymin, xmax, ymax), label)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Annotations for: {filename}")
    plt.show()

# Define paths
train_annotation_dir = "dataset/annotations/train"
train_image_base_dir = "dataset/Images/train"
val_annotation_dir = "dataset/annotations/val"
val_image_base_dir = "dataset/Images/val"

# Test with a random XML file from the train or val directory
print("Testing random train XML...")
test_random_xml(train_annotation_dir, train_image_base_dir)

print("Testing random val XML...")
test_random_xml(val_annotation_dir, val_image_base_dir)