import wandb
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDataset
from model import get_model

wandb.init(
    project="object-detection",
    config={
        "num_classes": 4,
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 0.005,
        "architecture": "FasterRCNN"
    }
)
config = wandb.config

dataset_path = "dataset"
num_classes = config.num_classes
num_epochs = config.num_epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
checkpoint_path = "model_checkpoint.pt"

transforms = ToTensor()
dataset = VOCDataset(dataset_path, transforms)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

start_epoch = 0
if checkpoint_path and torch.cuda.is_available():
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}...")

for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    tqdm_batches = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for images, targets in tqdm_batches:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        
        losses.backward()
        optimizer.step()

        tqdm_batches.set_postfix({"Batch Loss": losses.item()})
        wandb.log({"Batch Loss": losses.item()})
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    wandb.log({"Epoch Loss": epoch_loss, "Epoch": epoch + 1})

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    wandb.save(checkpoint_path)

wandb.finish()
