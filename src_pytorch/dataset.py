import os
import torch
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as F
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations = sorted(os.listdir(os.path.join(root, "annotations")))

        # Ensure only matching files are used
        self.images = [img for img in self.images if img.replace(".JPG", ".xml") in self.annotations]
        self.annotations = [ann for ann in self.annotations if ann.replace(".xml", ".JPG") in self.images]

        self.class_map = {"yellow": 1, "red": 2, "blue": 3}  # Class mapping

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, "images", self.images[idx])
            anno_path = os.path.join(self.root, "annotations", self.annotations[idx])

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Parse annotation
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            # Initialize boxes and labels
            boxes = []
            labels = []
            
            for obj in root.findall("object"):
                # Get the class name
                name = obj.find("name").text
                
                # Ensure the class name exists in the mapping
                if name not in self.class_map:
                    raise ValueError(f"Unexpected class name: {name}")
                
                # Map the class name to its numeric ID
                labels.append(self.class_map[name])

                # Parse bounding box
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}

            # Apply transforms, if any
            if self.transforms:
                img = self.transforms(img)

            return img, target
        except Exception as e:
            # Print a warning and skip this sample
            print(f"Warning: Skipping index {idx} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.images)
    