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
import cv2

warnings.filterwarnings('ignore')


class CHDDataset(Dataset):
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
            # Return dummy image
            dummy_image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0


def create_chd_sample_data():
    """Create sample CHD data"""
    print("Creating sample CHD classification data...")

    # Create directories
    os.makedirs('data/chd_classification/chd_positive', exist_ok=True)
    os.makedirs('data/chd_classification/chd_negative', exist_ok=True)

    # Create CHD Positive samples
    for i in range(30):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Abnormal patterns (asymmetrical)
        cv2.ellipse(img, (140, 150), (70, 55), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (260, 150), (45, 55), 0, 0, 360, (180, 180, 180), -1)

        cv2.putText(img, "CHD POSITIVE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/chd_classification/chd_positive/chd_pos_{i + 1:03d}.png', img)

    # Create CHD Negative samples
    for i in range(30):
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Normal patterns (symmetrical)
        cv2.ellipse(img, (150, 150), (55, 45), 0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (250, 150), (55, 45), 0, 0, 360, (180, 180, 180), -1)

        cv2.putText(img, "NORMAL HEART", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(img, f"Sample {i + 1}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f'data/chd_classification/chd_negative/chd_neg_{i + 1:03d}.png', img)

    print("Sample CHD data created!")


def load_chd_dataset(data_dir):
    """Load CHD classification dataset"""
    image_paths = []
    labels = []

    # CHD Positive cases
    chd_positive_dir = os.path.join(data_dir, 'chd_classification', 'chd_positive')
    if os.path.exists(chd_positive_dir):
        files = os.listdir(chd_positive_dir)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for img_file in image_files:
            image_paths.append(os.path.join(chd_positive_dir, img_file))
            labels.append(1)  # CHD Positive
    else:
        print(f"CHD positive directory not found: {chd_positive_dir}")

    # CHD Negative cases
    chd_negative_dir = os.path.join(data_dir, 'chd_classification', 'chd_negative')
    if os.path.exists(chd_negative_dir):
        files = os.listdir(chd_negative_dir)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for img_file in image_files:
            image_paths.append(os.path.join(chd_negative_dir, img_file))
            labels.append(0)  # CHD Negative
    else:
        print(f"CHD negative directory not found: {chd_negative_dir}")

    print(f"Loaded {len([x for x in labels if x == 1])} CHD positive images")
    print(f"Loaded {len([x for x in labels if x == 0])} CHD negative images")

    return image_paths, labels


def train_chd_classifier():
    # Configuration
    data_dir = 'data'
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.0001
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # First, check if data exists and create if needed
    chd_positive_dir = os.path.join(data_dir, 'chd_classification', 'chd_positive')
    chd_negative_dir = os.path.join(data_dir, 'chd_classification', 'chd_negative')

    if not os.path.exists(chd_positive_dir) or not os.path.exists(chd_negative_dir):
        print("CHD data directories not found. Creating sample data...")
        create_chd_sample_data()

    # Load dataset
    print("Loading CHD dataset...")
    image_paths, labels = load_chd_dataset(data_dir)

    if len(image_paths) == 0:
        print("ERROR: No CHD images found even after creating sample data!")
        return

    print(f"Training with {len(image_paths)} total images")

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

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

    # Create datasets
    train_dataset = CHDDataset(train_paths, train_labels, train_transform)
    val_dataset = CHDDataset(val_paths, val_labels, val_transform)

    # Handle class imbalance
    class_counts = np.bincount(train_labels)
    print(f"Class distribution: {class_counts}")

    if len(class_counts) == 2:
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = class_weights[train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("Initializing model...")
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
    if len(class_counts) == 2:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Starting CHD Classifier Training...")

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

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')

        # Save model
        torch.save(model.state_dict(), 'models/chd_classifier.pth')
        print(f'  Model saved to models/chd_classifier.pth')

    print("\nCHD Classifier training completed!")
    print(f"Final model saved as: models/chd_classifier.pth")

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
            plt.title('CHD Classifier Training Loss')

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Val Accuracy', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('CHD Classifier Validation Accuracy')

            plt.tight_layout()
            plt.savefig('chd_training_history.png')
            print("Training plot saved as chd_training_history.png")
        except Exception as e:
            print(f"Could not create plot: {e}")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    print("=== CHD Classifier Training ===")
    train_chd_classifier()