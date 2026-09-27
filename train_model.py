"""
HAND SLINGER - Hand Sign Detection Model Training (v2 - Balanced)
==================================================================
FIXES: Balanced training data across all classes to eliminate bias
toward over-represented classes (I_ME_MINE_MY, YOU, etc.)

Uses MediaPipe for hand landmark extraction + RandomForest/MLP classifier
Dataset: Frames_Word_Level (114 sign language words)
Target: 95%+ accuracy with BALANCED per-class performance

Techniques:
- MediaPipe Hands for hand landmark detection
- Feature normalization (relative to wrist position)
- BALANCED data augmentation (equal samples per class)
- Extra engineered features (distances, angles, curl ratios)
- RandomForest Classifier with optimized hyperparameters
- MLP Neural Network as secondary model
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import time

# ============== CONFIGURATION ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Frames_Word_Level")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_SAMPLES_PER_CLASS = 80   # Every class gets exactly this many samples
RANDOM_STATE = 42

# ============== MEDIAPIPE SETUP ==============
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.2
)


def extract_hand_landmarks(image_path):
    """
    Extract normalized hand landmarks from an image using MediaPipe.
    Tries multiple preprocessing approaches for maximum detection rate.
    Returns 126 features (2 hands x 21 landmarks x 3 coords), normalized.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    results = None
    # Try multiple image preprocessing approaches
    preprocessed_images = []
    
    # 1. Original
    preprocessed_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 2. Brightened
    bright = cv2.convertScaleAbs(image, alpha=1.4, beta=40)
    preprocessed_images.append(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
    
    # 3. Contrast enhanced
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    preprocessed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # 4. Resized
    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    preprocessed_images.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    for img_rgb in preprocessed_images:
        results = hands_detector.process(img_rgb)
        if results.multi_hand_landmarks:
            break

    if not results or not results.multi_hand_landmarks:
        return None

    all_landmarks = []

    for hand_landmarks in results.multi_hand_landmarks:
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Normalize relative to wrist (landmark 0)
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist

        # Scale normalization - max distance from wrist = 1
        max_dist = np.max(np.sqrt(np.sum(landmarks ** 2, axis=1)))
        if max_dist > 0:
            landmarks = landmarks / max_dist

        all_landmarks.extend(landmarks.flatten().tolist())

    # Pad to 126 (2 hands x 21 landmarks x 3 coords)
    while len(all_landmarks) < 126:
        all_landmarks.append(0.0)

    return all_landmarks[:126]


def compute_extra_features(landmarks_126):
    """
    Compute engineered features from landmarks for better accuracy.
    Must stay in sync with app.py's compute_extra_features.
    
    Features:
    - Pairwise fingertip distances (10)
    - Finger curl ratios (5)
    - Thumb-finger distances (4)
    - Finger spread angles (4)
    - Palm geometry (3)
    Total: 26 extra features
    """
    lm = np.array(landmarks_126[:63]).reshape(21, 3)

    # Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
    tips = [4, 8, 12, 16, 20]
    # MCP (knuckle) indices
    mcps = [2, 5, 9, 13, 17]
    # PIP (middle joint) indices
    pips = [3, 6, 10, 14, 18]

    extra = []

    # 1. Pairwise fingertip distances (10 features)
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dist = np.sqrt(np.sum((lm[tips[i]] - lm[tips[j]]) ** 2))
            extra.append(dist)

    # 2. Finger curl ratios: tip-to-wrist / mcp-to-wrist (5 features)
    for tip, mcp in zip(tips, mcps):
        tip_dist = np.sqrt(np.sum(lm[tip] ** 2))
        mcp_dist = np.sqrt(np.sum(lm[mcp] ** 2))
        curl = tip_dist / (mcp_dist + 1e-6)
        extra.append(curl)

    # 3. Thumb to each fingertip distance (4 features)
    for tip in tips[1:]:
        dist = np.sqrt(np.sum((lm[4] - lm[tip]) ** 2))
        extra.append(dist)

    # 4. Finger spread angles using dot product (4 features)
    for i in range(len(tips) - 1):
        v1 = lm[tips[i]] - lm[0]     # vector from wrist to tip i
        v2 = lm[tips[i+1]] - lm[0]   # vector from wrist to tip i+1
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        extra.append(cos_angle)

    # 5. Palm geometry (3 features)
    # Palm width: distance between index MCP and pinky MCP
    palm_width = np.sqrt(np.sum((lm[5] - lm[17]) ** 2))
    extra.append(palm_width)
    # Palm height: distance from wrist to middle finger MCP
    palm_height = np.sqrt(np.sum(lm[9] ** 2))
    extra.append(palm_height)
    # Aspect ratio
    extra.append(palm_width / (palm_height + 1e-6))

    return extra


def augment_landmarks_varied(landmarks, count):
    """
    Generate exactly 'count' augmented versions with diverse transformations.
    Each augmentation applies a random combination of transformations.
    """
    augmented = []
    landmarks = np.array(landmarks)

    for i in range(count):
        aug = landmarks.copy()
        
        # Random Gaussian noise (vary intensity)
        noise_std = np.random.uniform(0.01, 0.03)
        noise = np.random.normal(0, noise_std, size=aug.shape)
        aug = aug + noise

        # Random scale
        scale = np.random.uniform(0.88, 1.12)
        aug = aug * scale

        # Random 2D rotation
        angle = np.random.uniform(-0.2, 0.2)  # radians (~11 degrees)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for k in range(0, min(len(aug) - 2, 63), 3):
            x, y = aug[k], aug[k + 1]
            aug[k] = x * cos_a - y * sin_a
            aug[k + 1] = x * sin_a + y * cos_a

        # Random small translation (50% chance)
        if np.random.random() > 0.5:
            tx = np.random.uniform(-0.06, 0.06)
            ty = np.random.uniform(-0.06, 0.06)
            for k in range(0, min(len(aug) - 2, 63), 3):
                aug[k] += tx
                aug[k + 1] += ty

        # Mirror augmentation (30% chance)
        if np.random.random() > 0.7:
            for k in range(0, len(aug), 3):
                aug[k] = -aug[k]

        augmented.append(aug.tolist())

    return augmented


def load_dataset_balanced():
    """
    Load dataset with BALANCED augmentation.
    Every class gets exactly TARGET_SAMPLES_PER_CLASS samples,
    regardless of how many original images it has.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path not found: {DATASET_PATH}")
        sys.exit(1)

    class_dirs = sorted([d for d in os.listdir(DATASET_PATH)
                         if os.path.isdir(os.path.join(DATASET_PATH, d))])

    print(f"\nFound {len(class_dirs)} sign language classes")
    print(f"Target samples per class: {TARGET_SAMPLES_PER_CLASS}")
    print("=" * 70)

    CACHE_PATH = os.path.join(MODEL_DIR, "landmarks_cache.pkl")
    if os.path.exists(CACHE_PATH):
        print(f"Loading pre-extracted landmarks from cache ({CACHE_PATH})...")
        class_originals = joblib.load(CACHE_PATH)
        print(f"Loaded {len(class_originals)} classes from cache in 0.1s!")
    else:
        # Phase 1: Extract original landmarks per class
        class_originals = defaultdict(list)  # class_name -> list of landmark arrays
        total_images = 0
        total_detected = 0

        for idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(DATASET_PATH, class_name)
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            detected = 0
            for img_file in image_files:
                total_images += 1
                img_path = os.path.join(class_path, img_file)
                landmarks = extract_hand_landmarks(img_path)
                if landmarks is not None:
                    detected += 1
                    total_detected += 1
                    class_originals[class_name].append(landmarks)

            status = "OK" if detected > 0 else "SKIP"
            progress = f"[{idx + 1:3d}/{len(class_dirs)}]"
            print(f"  {progress} [{status:4s}] {class_name:<25s} : {detected}/{len(image_files)} detected")

        print(f"\n{'=' * 70}")
        print(f"  Images scanned    : {total_images}")
        print(f"  Hands detected    : {total_detected} ({total_detected / total_images * 100:.1f}%)")
        print(f"  Classes with data : {len(class_originals)}/{len(class_dirs)}")
        joblib.dump(class_originals, CACHE_PATH)

    # Phase 2: Balance all classes to TARGET_SAMPLES_PER_CLASS
    print(f"\n[Balancing] Generating {TARGET_SAMPLES_PER_CLASS} samples per class...")
    
    features = []
    labels = []
    
    for class_name, originals in class_originals.items():
        n_orig = len(originals)
        if n_orig == 0:
            continue

        # Add all original samples with features
        class_features = []
        for lm in originals:
            extra = compute_extra_features(lm)
            class_features.append(lm + extra)

        # Calculate how many augmented samples we need
        n_needed = TARGET_SAMPLES_PER_CLASS - n_orig
        
        if n_needed > 0:
            # Generate augmented samples, distributing evenly across originals
            aug_per_orig = n_needed // n_orig
            remainder = n_needed % n_orig
            
            for i, orig_lm in enumerate(originals):
                n_aug = aug_per_orig + (1 if i < remainder else 0)
                if n_aug > 0:
                    augmented = augment_landmarks_varied(orig_lm, n_aug)
                    for aug_lm in augmented:
                        aug_extra = compute_extra_features(aug_lm)
                        class_features.append(aug_lm + aug_extra)
        
        elif n_orig > TARGET_SAMPLES_PER_CLASS:
            # Subsample if too many originals (unlikely but handle it)
            indices = np.random.choice(n_orig, TARGET_SAMPLES_PER_CLASS, replace=False)
            class_features = [class_features[i] for i in indices]

        features.extend(class_features)
        labels.extend([class_name] * len(class_features))

    print(f"  Total balanced samples : {len(features)}")
    print(f"  Expected               : ~{TARGET_SAMPLES_PER_CLASS * len(class_originals)}")
    print(f"{'=' * 70}")

    return np.array(features), np.array(labels), class_dirs


def train_model():
    """Main training pipeline with balanced data."""
    print("\n" + "=" * 70)
    print("       HAND SLINGER - Model Training Pipeline v2")
    print("       BALANCED Training + Enhanced Features")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==================== PHASE 1: Balanced Data Loading ====================
    print("\n[PHASE 1/4] Loading dataset with BALANCED augmentation...")
    start_time = time.time()
    X, y, class_names = load_dataset_balanced()
    load_time = time.time() - start_time
    print(f"\nPhase 1 complete in {load_time:.1f}s")

    if len(X) == 0:
        print("\nERROR: No features could be extracted!")
        sys.exit(1)

    # ==================== PHASE 2: Data Preparation ====================
    print("\n[PHASE 2/4] Preparing data...")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Check class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    print(f"  Classes        : {len(unique)}")
    print(f"  Min samples    : {counts.min()}")
    print(f"  Max samples    : {counts.max()}")
    print(f"  Mean samples   : {counts.mean():.0f}")
    print(f"  Std samples    : {counts.std():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"  Training set   : {len(X_train)}")
    print(f"  Testing set    : {len(X_test)}")
    print(f"  Feature dim    : {X.shape[1]}")

    # ==================== PHASE 3: Model Training ====================
    print("\n[PHASE 3/4] Training classifiers on BALANCED data...")

    # --- Model 1: Random Forest ---
    print("\n  [Model 1] Random Forest Classifier")
    print("  " + "-" * 50)
    rf_model = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )

    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0

    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))

    print(f"  Train Accuracy   : {rf_train_acc * 100:.2f}%")
    print(f"  Test Accuracy    : {rf_accuracy * 100:.2f}%")
    print(f"  OOB Score        : {rf_model.oob_score_ * 100:.2f}%")
    print(f"  Training time    : {rf_time:.1f}s")

    # --- Model 2: MLP Neural Network ---
    print("\n  [Model 2] MLP Neural Network")
    print("  " + "-" * 50)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        alpha=0.0005
    )

    t0 = time.time()
    mlp_model.fit(X_train, y_train)
    mlp_time = time.time() - t0

    mlp_pred = mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    mlp_train_acc = accuracy_score(y_train, mlp_model.predict(X_train))

    print(f"  Train Accuracy   : {mlp_train_acc * 100:.2f}%")
    print(f"  Test Accuracy    : {mlp_accuracy * 100:.2f}%")
    print(f"  Iterations       : {mlp_model.n_iter_}")
    print(f"  Training time    : {mlp_time:.1f}s")

    # ==================== PHASE 4: Evaluation & Save ====================
    print("\n[PHASE 4/4] Evaluating and saving best model...")

    if rf_accuracy >= mlp_accuracy:
        best_model = rf_model
        best_accuracy = rf_accuracy
        best_name = "RandomForest"
        best_pred = rf_pred
    else:
        best_model = mlp_model
        best_accuracy = mlp_accuracy
        best_name = "MLP_NeuralNetwork"
        best_pred = mlp_pred

    print(f"\n  {'=' * 50}")
    print(f"  BEST MODEL: {best_name}")
    print(f"  ACCURACY  : {best_accuracy * 100:.2f}%")
    print(f"  {'=' * 50}")

    report_text = classification_report(
        y_test, best_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    report_dict = classification_report(
        y_test, best_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    print(f"\n  Classification Report:\n")
    print(report_text)

    # Check per-class performance
    low_perf = []
    for cls_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            if metrics['f1-score'] < 0.7 and cls_name not in ['accuracy', 'macro avg', 'weighted avg']:
                low_perf.append((cls_name, metrics['f1-score']))
    
    if low_perf:
        print(f"\n  Low-performing classes (F1 < 0.7):")
        for cls, f1 in sorted(low_perf, key=lambda x: x[1]):
            print(f"    - {cls}: F1={f1:.2f}")

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, "hand_sign_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    class_names_path = os.path.join(MODEL_DIR, "class_names.json")
    report_path = os.path.join(MODEL_DIR, "training_report.json")

    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, encoder_path)

    valid_classes = list(label_encoder.classes_)
    with open(class_names_path, 'w') as f:
        json.dump(valid_classes, f, indent=2)

    training_report = {
        "best_model": best_name,
        "test_accuracy": round(best_accuracy, 4),
        "rf_test_accuracy": round(rf_accuracy, 4),
        "mlp_test_accuracy": round(mlp_accuracy, 4),
        "num_classes": len(valid_classes),
        "total_samples": len(X),
        "target_per_class": TARGET_SAMPLES_PER_CLASS,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_dimensions": int(X.shape[1]),
        "classification_report": report_dict,
        "classes": valid_classes
    }

    with open(report_path, 'w') as f:
        json.dump(training_report, f, indent=2)

    print(f"\n  Saved artifacts:")
    print(f"    Model          : {model_path}")
    print(f"    Label encoder  : {encoder_path}")
    print(f"    Class names    : {class_names_path}")
    print(f"    Report         : {report_path}")

    if best_accuracy >= 0.95:
        print(f"\n  [PASS] Target accuracy 95% ACHIEVED: {best_accuracy * 100:.2f}%")
    elif best_accuracy >= 0.90:
        print(f"\n  [OK] Accuracy above 90%: {best_accuracy * 100:.2f}%")
    else:
        print(f"\n  [INFO] Accuracy: {best_accuracy * 100:.2f}%")

    print(f"\n{'=' * 70}")
    print(f"  Training pipeline v2 complete!")
    print(f"{'=' * 70}\n")

    return best_model, label_encoder


if __name__ == "__main__":
    train_model()
