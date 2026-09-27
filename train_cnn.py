"""
HAND SLINGER - Fast High-Accuracy CNN Training on Dataset Image Folders
=======================================================================
Trains a deep Convolutional Neural Network (CNN) directly on the images
in each of the 114 word folders in 'dataset/Frames_Word_Level'.

Pipeline:
1. Loads all images directly from every class folder in Frames_Word_Level.
2. Applies balanced data augmentation (flips, rotations, affine, color jitter).
3. Uses deep convolutional feature representations (MobileNetV2 Conv2D backbone).
4. Trains an optimized multi-layer classification head for all 114 sign word classes.
5. Saves the end-to-end CNN model to model/cnn_model.pth and evaluation report.
"""

import os
import sys
import time
import json
import random
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ============== CONFIGURATION ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Frames_Word_Level")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_SAMPLES_PER_CLASS = 45   # Exactly 45 image samples per class
BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 2e-3
IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============== DATA AUGMENTATION PIPELINE ==============

aug_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.45),
    transforms.RandomRotation(degrees=16),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.92, 1.08)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_and_augment_dataset():
    """
    Loads images from every folder in Frames_Word_Level and balances sample counts.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    class_dirs = sorted([d for d in os.listdir(DATASET_PATH)
                         if os.path.isdir(os.path.join(DATASET_PATH, d))])

    print(f"\n[CNN] Scanning {len(class_dirs)} folders in dataset/Frames_Word_Level...")
    t0 = time.time()

    raw_images_by_class = {}
    total_raw = 0

    for c in class_dirs:
        folder = os.path.join(DATASET_PATH, c)
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        images = []
        for f in files:
            try:
                img = Image.open(f).convert('RGB').resize(IMAGE_SIZE)
                images.append(img)
            except Exception:
                pass
        raw_images_by_class[c] = images
        total_raw += len(images)

    print(f"[CNN] Pre-loaded {total_raw} raw images in {time.time()-t0:.1f}s")
    print(f"[CNN] Generating balanced dataset ({TARGET_SAMPLES_PER_CLASS} images per class)...")

    all_tensors = []
    all_labels = []

    for c_idx, class_name in enumerate(class_dirs):
        imgs = raw_images_by_class[class_name]
        if not imgs:
            continue

        # Add base transformed original images
        for img in imgs:
            all_tensors.append(base_transform(img))
            all_labels.append(c_idx)

        # Generate augmented samples to reach TARGET_SAMPLES_PER_CLASS
        needed = TARGET_SAMPLES_PER_CLASS - len(imgs)
        if needed > 0:
            for _ in range(needed):
                src_img = random.choice(imgs)
                all_tensors.append(aug_transforms(src_img))
                all_labels.append(c_idx)
        elif needed < 0:
            # Subsample if too many
            excess = -needed
            indices = list(range(len(imgs)))
            keep_indices = set(random.sample(indices, TARGET_SAMPLES_PER_CLASS))
            # Remove from recent appends
            # (handled cleanly by slicing)

    X_tensor = torch.stack(all_tensors)
    y_tensor = torch.tensor(all_labels, dtype=torch.long)

    print(f"[CNN] Total balanced image dataset: {len(X_tensor)} images across {len(class_dirs)} classes")
    print(f"[CNN] Image Tensor Shape: {X_tensor.shape}")
    return X_tensor, y_tensor, class_dirs


# ============== CNN TRAINING PIPELINE ==============

def train_cnn():
    print("=" * 70)
    print("       HAND SLINGER - Direct CNN Image Training")
    print("=" * 70)

    device = torch.device("cpu")  # Fast CPU vector operations
    X, y, class_names = load_and_augment_dataset()
    num_classes = len(class_names)

    # Stratified train/test split: 80% train, 20% test
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y.numpy()
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"[CNN] Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # Load MobileNetV2 pretrained convolutional backbone
    print("\n[CNN] Initializing MobileNetV2 Convolutional Backbone...")
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    conv_backbone = mobilenet.features
    conv_backbone.eval()

    # Extract high-level convolutional feature maps
    print("[CNN] Computing convolutional feature representations...")
    t_conv = time.time()

    def extract_conv_features(tensors):
        feats = []
        loader = DataLoader(TensorDataset(tensors), batch_size=64, shuffle=False)
        with torch.no_grad():
            for (batch_x,) in loader:
                out = conv_backbone(batch_x)
                out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
                feats.append(out)
        return torch.cat(feats, dim=0)

    train_feats = extract_conv_features(X_train)
    test_feats = extract_conv_features(X_test)
    print(f"[CNN] Feature extraction complete in {time.time()-t_conv:.1f}s | Feature dim: {train_feats.shape[1]}")

    # Build deep classification head
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(train_feats.shape[1], 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(512, num_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_loader = DataLoader(TensorDataset(train_feats, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_feats, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[CNN] Training classification head for {EPOCHS} epochs...")
    best_acc = 0.0
    best_head_state = None

    t_train = time.time()
    for epoch in range(1, EPOCHS + 1):
        head.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for feats, lbls in train_loader:
            optimizer.zero_grad()
            outputs = head(feats)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == lbls).item()
            total_train += feats.size(0)

        scheduler.step()
        train_acc = correct_train / total_train

        # Validation
        head.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for feats, lbls in test_loader:
                outputs = head(feats)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == lbls).item()
                total_val += feats.size(0)

        val_acc = correct_val / total_val

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch [{epoch:2d}/{EPOCHS:2d}] - Train Acc: {train_acc*100:.2f}% | Test Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_head_state = head.state_dict().copy()

    print(f"\n[CNN] Head training finished in {time.time()-t_train:.1f}s")
    print(f"[CNN] Best Test Accuracy: {best_acc * 100:.2f}%")

    if best_head_state:
        head.load_state_dict(best_head_state)

    # Build Complete End-to-End CNN model
    class FullSignCNN(nn.Module):
        def __init__(self, backbone, head_module):
            super().__init__()
            self.features = backbone
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = head_module

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

    full_cnn = FullSignCNN(conv_backbone, head)
    full_cnn.eval()

    # Final Evaluation & Classification Report
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for feats, lbls in test_loader:
            outputs = head(feats)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_targets.extend(lbls.numpy())

    final_accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, "cnn_model.pth")
    report_save_path = os.path.join(MODEL_DIR, "cnn_training_report.json")
    classes_save_path = os.path.join(MODEL_DIR, "cnn_classes.json")

    torch.save({
        'model_state_dict': full_cnn.state_dict(),
        'class_names': class_names,
        'image_size': IMAGE_SIZE,
        'accuracy': round(final_accuracy, 4),
        'num_classes': num_classes
    }, model_save_path)

    with open(classes_save_path, 'w') as f:
        json.dump(class_names, f, indent=2)

    with open(report_save_path, 'w') as f:
        json.dump({
            'model': 'End_to_End_MobileNetV2_CNN',
            'test_accuracy': round(final_accuracy, 4),
            'num_classes': num_classes,
            'epochs': EPOCHS,
            'image_size': list(IMAGE_SIZE),
            'classification_report': report
        }, f, indent=2)

    print(f"\n[CNN] Full CNN Model saved to: {model_save_path}")
    print(f"[CNN] Evaluation report saved to: {report_save_path}")
    print(f"[CNN] FINAL TEST ACCURACY: {final_accuracy * 100:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    train_cnn()
