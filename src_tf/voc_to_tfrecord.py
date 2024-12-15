import os
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

def create_tf_example(annotation_path, images_base_path, label_map):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    filename = root.find('filename').text
    class_name = root.find('object/name').text 
    class_folder = class_name.lower() 
    img_path = os.path.join(images_base_path, class_folder, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image = fid.read()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        classes_text.append(label.encode('utf8'))
        classes.append(label_map[label]) 

        bndbox = obj.find('bndbox')
        xmin.append(float(bndbox.find('xmin').text) / width)
        ymin.append(float(bndbox.find('ymin').text) / height)
        xmax.append(float(bndbox.find('xmax').text) / width)
        ymax.append(float(bndbox.find('ymax').text) / height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(output_path, annotations_dir, images_base_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    for annotation_file in os.listdir(annotations_dir):
        if not annotation_file.endswith('.xml'):
            continue

        annotation_path = os.path.join(annotations_dir, annotation_file)
        tf_example = create_tf_example(annotation_path, images_base_path, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"TFRecord successfully created at {output_path}")


if __name__ == "__main__":
    train_annotations_dir = "dataset/annotations/train"
    val_annotations_dir = "dataset/annotations/val"
    train_images_base_path = "dataset/Images/train"
    val_images_base_path = "dataset/Images/val"
    train_tfrecord_path = "dataset/annotations/train.tfrecord"
    val_tfrecord_path = "dataset/annotations/val.tfrecord"

    label_map = {
        'red': 1,
        'blue': 2,
        'yellow': 3
    }

    print("Generating Train TFRecord...")
    create_tf_record(train_tfrecord_path, train_annotations_dir, train_images_base_path, label_map)

    print("Generating Val TFRecord...")
    create_tf_record(val_tfrecord_path, val_annotations_dir, val_images_base_path, label_map)
