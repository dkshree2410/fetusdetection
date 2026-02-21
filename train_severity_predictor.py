import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SeverityDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            dummy_image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0


def load_severity_dataset(data_dir):
    """Load CHD severity dataset"""
    image_paths = []
    labels = []
    severity_mapping = {'mild': 0, 'moderate': 1, 'severe': 2}

    for severity_level, label in severity_mapping.items():
        severity_dir = os.path.join(data_dir, 'severity_data', severity_level)
        if os.path.exists(severity_dir):
            files = [f for f in os.listdir(severity_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in files:
                image_paths.append(os.path.join(severity_dir, img_file))
                labels.append(label)

            print(f"Loaded {len(files)} {severity_level} cases")
        else:
            print(f"Warning: {severity_dir} not found")

    return image_paths, labels


def train_severity_predictor():
    # Configuration
    data_dir = 'data'
    batch_size = 8
    num_epochs = 15
    learning_rate = 0.0001
    num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("Loading severity dataset...")
    image_paths, labels = load_severity_dataset(data_dir)

    if len(image_paths) == 0:
        print("ERROR: No severity data found!")
        print("Please run create_severity_data.py first")
        return

    print(f"Loaded {len(image_paths)} total severity images")

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    # Create datasets
    train_dataset = SeverityDataset(train_paths, train_labels, train_transform)
    val_dataset = SeverityDataset(val_paths, val_labels, val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("Initializing severity predictor model...")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Starting Severity Predictor Training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')

        # Save model
        torch.save(model.state_dict(), 'models/severity_predictor.pth')
        print(f'  Model saved to models/severity_predictor.pth')

    print("\nSeverity Predictor training completed!")
    print(f"Final model saved as: models/severity_predictor.pth")

    # Plot training history
    if num_epochs > 1:
        try:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Severity Predictor Training Loss')

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Severity Predictor Validation Accuracy')

            plt.tight_layout()
            plt.savefig('severity_training_history.png')
            print("Training plot saved as severity_training_history.png")
        except Exception as e:
            print(f"Could not create plot: {e}")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    print("=== CHD Severity Predictor Training ===")
    train_severity_predictor()