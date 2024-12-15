
# **FTC Vision**

**FTC Vision** is an object detection project designed for the **2024-2025 FIRST Tech Challenge (FTC)** season. This repository provides both **PyTorch** and **TensorFlow** implementations, enabling flexible training, validation, and inference workflows.

## **Key Features**
- **Multi-Framework Support**: Implementations in both PyTorch and TensorFlow for training and inference.
- **Dataset**: Includes annotations and images of FTC game pieces. Available in VOC format and as TensorFlow-ready TFRecord files.
- **TFLite Export**: TensorFlow models can be exported to TFLite for deployment on lightweight devices.
- **Comprehensive Tools**: Utilities for preprocessing, dataset generation, and model conversion between frameworks.

## **Dataset**
The dataset used in this project is hosted on **Hugging Face** and is accessible at the link below:


| **Resources**   | **Description**                                                                                                                                     |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [FTC Vision](https://huggingface.co/datasets/torinriley/FTCVision) | Annotated dataset in VOC format, split into train/val with subdirectories for each class. Includes train/val TFRecord files and a label map. |
| [FTC Vision - PyTorch](https://huggingface.co/torinriley/FTCVision-PyTorch) | PyTorch implementation of the FTC Vision model, including training scripts and model weights. |


## **Repository Structure**
```plaintext
.
├── pytorch/                      # PyTorch implementation
├── tensorflow/                   # TensorFlow implementation
├── models/                       # Pretrained and fine-tuned models
├── datasets/                     # Dataset management and preprocessing scripts
├── utils/                        # Utility scripts for model training and evaluation
├── README.md                     # Project overview
└── requirements.txt              # Dependencies for the project
