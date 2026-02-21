# train_fetal_validator.py - COMPLETE WORKING VERSION
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import cv2

warnings.filterwarnings('ignore')


class FetalUltrasoundDataset(Dataset):
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
            print(f"Error loading image {image_path}: {e}")
            dummy_image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0


def load_dataset(data_dir):
    """Load BOTH fetal ultrasound AND non-fetal images"""
    image_paths = []
    labels = []

    # Fetal Ultrasound Images (Label 1)
    fetal_dir = os.path.join(data_dir, 'fetal_ultrasound')
    if os.path.exists(fetal_dir):
        for img_file in os.listdir(fetal_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(fetal_dir, img_file))
                labels.append(1)  # Fetal ultrasound = 1
                print(f"✓ Fetal: {img_file}")

    # Non-Fetal Images (Label 0) - SUNSET, ROSE, etc.
    non_fetal_dir = os.path.join(data_dir, 'non_fetal')
    if os.path.exists(non_fetal_dir):
        for img_file in os.listdir(non_fetal_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(non_fetal_dir, img_file))
                labels.append(0)  # Non-fetal = 0
                print(f"✗ Non-fetal: {img_file}")
    else:
        # Create non_fetal folder if not exists
        os.makedirs(non_fetal_dir, exist_ok=True)
        print("⚠️ Created non_fetal folder - please add normal images there")

    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Fetal ultrasound images: {labels.count(1)}")
    print(f"   Non-fetal images: {labels.count(0)}")
    print(f"   Total images: {len(image_paths)}")

    return image_paths, labels


def create_quick_non_fetal_samples():
    """Quickly create some non-fetal samples if folder is empty"""
    non_fetal_dir = 'data/non_fetal'

    if len(os.listdir(non_fetal_dir)) == 0:
        print("🔄 Creating quick non-fetal samples...")

        # Create 10 simple non-fetal images
        for i in range(10):
            img = np.zeros((224, 224, 3), dtype=np.uint8)

            if i % 3 == 0:
                # Blue sky
                img[:, :] = [135, 206, 235]  # Sky blue
                cv2.putText(img, "SKY", (80, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif i % 3 == 1:
                # Green grass
                img[:, :] = [34, 139, 34]  # Forest green
                cv2.putText(img, "GRASS", (70, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Red object
                img[:, :] = [220, 20, 60]  # Crimson red
                cv2.putText(img, "OBJECT", (60, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            filename = os.path.join(non_fetal_dir, f"sample_{i + 1}.png")
            cv2.imwrite(filename, img)
            print(f"Created: {filename}")

        print("✅ Created 10 non-fetal samples")


def train_fetal_validator():
    # Configuration
    data_dir = 'data'
    batch_size = 8  # Small batch for stability
    num_epochs = 5  # Only 5 epochs for quick testing
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🖥️ Using device: {device}")

    # Create directories
    os.makedirs('data/fetal_ultrasound', exist_ok=True)
    os.makedirs('data/non_fetal', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("📁 Loading dataset...")
    image_paths, labels = load_dataset(data_dir)

    # Create quick samples if no non-fetal images
    if labels.count(0) == 0:
        create_quick_non_fetal_samples()
        image_paths, labels = load_dataset(data_dir)

    # Check if we have both classes
    if labels.count(0) == 0 or labels.count(1) == 0:
        print("❌ ERROR: Need both fetal AND non-fetal images!")
        print("   Please add some normal images to: data/non_fetal/")
        return

    print(f"✅ Dataset ready: {len(image_paths)} images")

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"📊 Training samples: {len(train_paths)}")
    print(f"📊 Validation samples: {len(val_paths)}")

    # Create datasets
    train_dataset = FetalUltrasoundDataset(train_paths, train_labels, train_transform)
    val_dataset = FetalUltrasoundDataset(val_paths, val_labels, train_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print("🎯 Initializing model...")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("🚀 Starting training...\n")

    # Training history
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            images = images.to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images).squeeze()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'✅ Epoch [{epoch + 1}/{num_epochs}]')
        print(f'   Train Loss: {train_loss:.4f}')
        print(f'   Val Accuracy: {val_accuracy:.2f}%')

    # Save model
    torch.save(model.state_dict(), 'models/fetal_resnet18.pth')
    print(f'💾 Model saved to: models/fetal_resnet18.pth')

    # Test the model
    test_trained_model(model, device, train_transform)


def test_trained_model(model, device, transform):
    """Test the trained model on sample images"""
    print("\n🧪 TESTING TRAINED MODEL...")

    model.eval()

    # Test on sample images from both classes
    test_images = []

    # Get one fetal image
    fetal_files = os.listdir('data/fetal_ultrasound')
    if fetal_files:
        test_images.append(('data/fetal_ultrasound/' + fetal_files[0], 'REAL FETAL'))

    # Get one non-fetal image
    non_fetal_files = os.listdir('data/non_fetal')
    if non_fetal_files:
        test_images.append(('data/non_fetal/' + non_fetal_files[0], 'REAL NON-FETAL'))

    print("📊 PREDICTION RESULTS:")
    for img_path, true_label in test_images:
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor).item()
                prediction = "FETAL" if output > 0.5 else "NON-FETAL"
                confidence = output if output > 0.5 else 1 - output

            print(f"📸 {os.path.basename(img_path)}")
            print(f"   True: {true_label}")
            print(f"   Predicted: {prediction}")
            print(f"   Confidence: {confidence:.4f}")
            print()

        except Exception as e:
            print(f"❌ Error testing {img_path}: {e}")


if __name__ == '__main__':
    print("=== 🎯 FETAL ULTRASOUND VALIDATOR ===")
    print("This model distinguishes between fetal ultrasound and normal images")
    train_fetal_validator()