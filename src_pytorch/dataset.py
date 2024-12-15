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

        self.images = [img for img in self.images if img.replace(".JPG", ".xml") in self.annotations]
        self.annotations = [ann for ann in self.annotations if ann.replace(".xml", ".JPG") in self.images]

        self.class_map = {"yellow": 1, "red": 2, "blue": 3}

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, "images", self.images[idx])
            anno_path = os.path.join(self.root, "annotations", self.annotations[idx])

            img = Image.open(img_path).convert("RGB")

            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            boxes = []
            labels = []
            
            for obj in root.findall("object"):
                name = obj.find("name").text
                
                if name not in self.class_map:
                    raise ValueError(f"Unexpected class name: {name}")
                
                labels.append(self.class_map[name])

                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}

            if self.transforms:
                img = self.transforms(img)

            return img, target
        except Exception as e:
            print(f"Warning: Skipping index {idx} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.images)
    
