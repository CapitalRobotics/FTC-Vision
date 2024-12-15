#!/bin/bash

DATASET_URL="https://huggingface.co/datasets/torinriley/FTCVision"

TARGET_DIR="dataset"

if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is not installed. Please install it and try again."
    exit 1
fi

echo "Cloning the FTCVision dataset into $TARGET_DIR..."
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR already exists. Deleting it..."
    rm -rf "$TARGET_DIR"
fi

git lfs install
git clone "$DATASET_URL" "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "Dataset successfully cloned into $TARGET_DIR."
else
    echo "Error cloning the dataset. Please check your internet connection or the dataset URL."
    exit 1
fi
