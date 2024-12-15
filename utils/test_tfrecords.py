import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

def parse_tfrecord(tfrecord):
    """
    Parses a single TFRecord into image and bounding box data.

    Args:
        tfrecord: A serialized TFRecord example.

    Returns:
        A tuple containing the image and bounding boxes.
    """
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }

    example = tf.io.parse_single_example(tfrecord, feature_description)

    # Decode image
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    filename = example['image/filename'].numpy().decode("utf-8")

    # Decode bounding boxes
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin']).numpy()
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax']).numpy()
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin']).numpy()
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax']).numpy()
    class_text = tf.sparse.to_dense(example['image/object/class/text']).numpy()

    bboxes = []
    for i in range(len(xmin)):
        bboxes.append((xmin[i], ymin[i], xmax[i], ymax[i], class_text[i].decode("utf-8")))

    return image, filename, bboxes

def draw_bounding_boxes(image, bboxes):
    """
    Draw bounding boxes on an image.

    Args:
        image: A numpy array representing the image.
        bboxes: A list of bounding boxes and class labels.
    """
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, label = bbox
        xmin, ymin, xmax, ymax = int(xmin * image.shape[1]), int(ymin * image.shape[0]), int(xmax * image.shape[1]), int(ymax * image.shape[0])

        # Draw the bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        # Add the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        cv2.rectangle(image, (xmin, ymin - 20), (xmin + text_size[0], ymin), color, -1)
        cv2.putText(image, label, (xmin, ymin - 5), font, font_scale, (255, 255, 255), 1)

def visualize_tfrecord(tfrecord_path, num_examples=5):
    """
    Visualizes examples from a TFRecord file.

    Args:
        tfrecord_path (str): Path to the TFRecord file.
        num_examples (int): Number of examples to visualize.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for i, raw_record in enumerate(dataset):
        example = parse_tfrecord(raw_record)
        image, filename, bboxes = example

        # Convert the image to numpy
        image = image.numpy()

        # Draw bounding boxes
        draw_bounding_boxes(image, bboxes)

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Example: {filename}")
        plt.show()

        if i + 1 >= num_examples:
            break

if __name__ == "__main__":
    # Paths to the TFRecord files
    train_tfrecord = "dataset/annotations/train.tfrecord"
    val_tfrecord = "dataset/annotations/val.tfrecord"

    print("Visualizing Train TFRecord...")
    visualize_tfrecord(train_tfrecord, num_examples=3)

    print("Visualizing Val TFRecord...")
    visualize_tfrecord(val_tfrecord, num_examples=3)