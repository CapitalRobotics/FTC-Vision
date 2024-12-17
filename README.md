
# **FTC Vision**

**FTC Vision** is an object detection project designed for the **2024-2025 FIRST Tech Challenge (FTC)** season. This repository provides both **PyTorch** and **TensorFlow** implementations, enabling flexible training, validation, and inference workflows.

## **Key Features**
- **Multi-Framework Support**: Implementations in both PyTorch and TensorFlow for training and inference.
- **Dataset**: Includes annotations and images of FTC game pieces. Available in VOC format and as TensorFlow-ready TFRecord files.
- **TFLite Export**: TensorFlow models can be exported to TFLite for deployment on lightweight devices.
- **Comprehensive Tools**: Utilities for preprocessing, dataset generation, and model conversion between frameworks.

## **Resources**
The Resources used in this project is hosted on **Hugging Face** and is accessible at the links below:


| **Resources**   | **Description**                                                                                                                                     |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [FTC Vision](https://huggingface.co/datasets/torinriley/FTCVision) | Annotated dataset in VOC format, split into train/val with subdirectories for each class. Includes train/val TFRecord files and a label map. |
| [FTC Vision - PyTorch](https://huggingface.co/torinriley/FTCVision-PyTorch) | PyTorch implementation of the FTC Vision model, including training scripts and model weights. |
| [Training Docs](https://huggingface.co/torinriley/FTCVision-PyTorch/tree/main/DOCS) |  Complete documentation for training of the PyTorch implimentaiton of FTC Vision|



## **Repository Structure**
```plaintext
.
├── DOCS/                         # Repository documentation
├── src_pytorch/                  # PyTorch implementation
├── src_tf/                       # TensorFlow implementation
├── utils/                        # Utility scripts for model training and evaluation
├── README.md                     # Project overview
└── requirements.txt              # Dependencies for the project
```

## Start Here
Start by setting up your development environment:

[Environment Setup](https://github.com/CapitalRobotics/ObjectDetecion/blob/main/DOCS/setup.md)

[Demo Notebook](https://github.com/CapitalRobotics/ObjectDetecion/blob/main/src_pytorch/demo.ipynb)

[PyTorch Model Archetecture](https://huggingface.co/torinriley/FTCVision-PyTorch/blob/main/DOCS/Archetecture.md)


