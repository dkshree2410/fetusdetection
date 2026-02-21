import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.encoder.append(UNet._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = UNet._block(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(UNet._block(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return torch.sigmoid(self.final_conv(x))


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Load mask
            mask = Image.open(mask_path).convert('L')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0.5).float()  # Binarize mask

            return image, mask

        except Exception as e:
            print(f"Error loading {image_path} or {mask_path}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, 256, 256)
            dummy_mask = torch.zeros(1, 256, 256)
            return dummy_image, dummy_mask


def load_segmentation_dataset(data_dir):
    """Load segmentation dataset with images and masks"""
    image_paths = []
    mask_paths = []

    images_dir = os.path.join(data_dir, 'fetal_ultrasound')
    masks_dir = os.path.join(data_dir, 'segmentation_masks')

    print(f"Looking for images in: {images_dir}")
    print(f"Looking for masks in: {masks_dir}")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return image_paths, mask_paths

    if not os.path.exists(masks_dir):
        print(f"Error: Masks directory not found: {masks_dir}")
        return image_paths, mask_paths

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} image files")

    for img_file in image_files:
        image_path = os.path.join(images_dir, img_file)

        # Look for corresponding mask
        base_name = os.path.splitext(img_file)[0]
        mask_file = base_name + '_mask.png'
        mask_path = os.path.join(masks_dir, mask_file)

        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            # Try other mask naming conventions
            for ext in ['.png', '.jpg', '.jpeg']:
                mask_file = base_name + '_mask' + ext
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    image_paths.append(image_path)
                    mask_paths.append(mask_path)
                    break
            else:
                print(f"Warning: No mask found for {img_file}")

    print(f"Loaded {len(image_paths)} image-mask pairs for segmentation")
    return image_paths, mask_paths


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred, target, smooth=1e-6):
    """Calculate Intersection over Union"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice


def train_segmentation_model():
    # Configuration
    data_dir = 'data'
    batch_size = 4
    num_epochs = 20
    learning_rate = 0.001
    image_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load dataset
    print("Loading segmentation dataset...")
    image_paths, mask_paths = load_segmentation_dataset(data_dir)

    if len(image_paths) == 0:
        print("ERROR: No segmentation data found!")
        print("Please run create_segmentation_data.py first")
        return

    print(f"Successfully loaded {len(image_paths)} image-mask pairs")

    # Split dataset
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(train_image_paths)}")
    print(f"Validation samples: {len(val_image_paths)}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = SegmentationDataset(
        train_image_paths, train_mask_paths, train_transform, mask_transform
    )
    val_dataset = SegmentationDataset(
        val_image_paths, val_mask_paths, val_transform, mask_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print("Initializing U-Net model...")
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # FIXED: Remove verbose parameter from scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training history
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []
    best_dice = 0

    print("Starting Segmentation Model Training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 2 == 0:
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        iou_score_val = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate metrics
                pred_masks = (outputs > 0.5).float()
                dice_score += dice_coefficient(pred_masks, masks).item()
                iou_score_val += iou_score(pred_masks, masks).item()

        val_loss /= len(val_loader)
        dice_score /= len(val_loader)
        iou_score_val /= len(val_loader)

        val_losses.append(val_loss)
        val_dice_scores.append(dice_score)
        val_iou_scores.append(iou_score_val)

        # Update learning rate
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Dice: {dice_score:.4f}')
        print(f'  IoU: {iou_score_val:.4f}')

        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), 'models/unet_segmentation.pth')
            print(f'  ✓ Best model saved with Dice: {dice_score:.4f}')
        else:
            # Still save the model every epoch for safety
            torch.save(model.state_dict(), 'models/unet_segmentation.pth')
            print(f'  Model saved')

    print("\nSegmentation training completed!")
    print(f"Final model saved as: models/unet_segmentation.pth")
    print(f"Best Dice score: {best_dice:.4f}")

    # Plot training history
    plot_training_history(train_losses, val_losses, val_dice_scores, val_iou_scores)

    # Final evaluation
    evaluate_final_model(model, val_loader, device)


def plot_training_history(train_losses, val_losses, val_dice_scores, val_iou_scores):
    """Plot training history"""
    try:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Segmentation Training Loss')

        plt.subplot(1, 3, 2)
        plt.plot(val_dice_scores, label='Dice Score', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.legend()
        plt.title('Validation Dice Score')

        plt.subplot(1, 3, 3)
        plt.plot(val_iou_scores, label='IoU Score', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.title('Validation IoU Score')

        plt.tight_layout()
        plt.savefig('segmentation_training_history.png')
        print("Training plot saved as segmentation_training_history.png")
    except Exception as e:
        print(f"Could not create plot: {e}")


def evaluate_final_model(model, val_loader, device):
    """Final evaluation of the trained model"""
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred_masks = (outputs > 0.5).float()

            batch_dice = dice_coefficient(pred_masks, masks).item()
            batch_iou = iou_score(pred_masks, masks).item()

            total_dice += batch_dice * images.size(0)
            total_iou += batch_iou * images.size(0)
            total_samples += images.size(0)

    final_dice = total_dice / total_samples
    final_iou = total_iou / total_samples

    print(f"\nFinal Evaluation Results:")
    print(f"Dice Coefficient: {final_dice:.4f}")
    print(f"IoU Score: {final_iou:.4f}")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    print("=== U-Net Segmentation Model Training ===")
    train_segmentation_model()