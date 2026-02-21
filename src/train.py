import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import parse_labels, calculate_class_weights, get_available_images
from dataset import MultiLabelDataset
from model import MultiLabelModel
from plot_loss import plot_training_loss


def create_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform


def masked_bce_loss(logits, labels, masks, pos_weights=None):

    bce_loss_fn = nn.BCEWithLogitsLoss(weight=pos_weights, reduction='none')
    loss = bce_loss_fn(logits, labels)
    masked_loss = loss * masks
    num_valid = masks.sum() + 1e-8
    return masked_loss.sum() / num_valid


def train(model, train_loader, optimizer, pos_weights, device, num_epochs=10):
    model.to(device)
    model.train()
    
    loss_history = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            loss = masked_bce_loss(logits, labels, masks, pos_weights=pos_weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loss_history.append(loss.item())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"  Average epoch loss: {avg_epoch_loss:.4f}")
    
    return loss_history


def main():
    IMAGES_DIR = Path(__file__).parent.parent / 'images'
    LABELS_PATH = Path(__file__).parent.parent / 'labels.txt'
    MODEL_SAVE_PATH = Path(__file__).parent.parent / 'model.pth'
    LOSS_PLOT_PATH = Path(__file__).parent.parent / 'loss_plot.png'
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nParsing labels...")
    labels_dict = parse_labels(LABELS_PATH)
    print(f"Total images in labels: {len(labels_dict)}")
    
    available_images = get_available_images(labels_dict, IMAGES_DIR)
    print(f"Available images in folder: {len(available_images)}")
    
    print("\nCalculating class weights...")
    pos_weights = calculate_class_weights(labels_dict, NUM_CLASSES)
    print(f"Weights: {pos_weights}")
    
    print("\nCreating dataset...")
    transform = create_transforms()
    dataset = MultiLabelDataset(available_images, labels_dict, IMAGES_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches: {len(train_loader)}")
    
    print("\nCreating model...")
    model = MultiLabelModel(num_classes=NUM_CLASSES)
    print("ResNet18 loaded")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting training...")
    loss_history = train(
        model,
        train_loader,
        optimizer,
        pos_weights.to(device),
        device,
        num_epochs=NUM_EPOCHS
    )
    
    print(f"\nSaving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Saved!")
    
    print(f"Generating loss plot...")
    plot_training_loss(loss_history, LOSS_PLOT_PATH)
    
    print("\nDone!")



if __name__ == '__main__':
    main()
