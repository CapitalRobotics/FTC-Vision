import os

val_dir = "dataset/Images/val"
val_images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

if not val_images:
    print(f"No images found in {val_dir}")
else:
    print(f"Found {len(val_images)} images in {val_dir}:")
    print(val_images)
    
