#!/bin/bash

MODEL_URL="https://huggingface.co/torinriley/FTCVision-PyTorch"
TARGET_DIR="../models"

if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is not installed. Please install it and try again."
    exit 1
fi

echo "Cloning the FTCVision-PyTorch model into $TARGET_DIR..."
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR already exists. Deleting it..."
    rm -rf "$TARGET_DIR"
fi

git lfs install
git clone "$MODEL_URL" "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "Model successfully cloned into $TARGET_DIR."
else
    echo "Error cloning the model. Please check your internet connection or the model URL."
    exit 1
fi
