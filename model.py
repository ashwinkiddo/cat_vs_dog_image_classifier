# ============================================================
# Cat vs Dog Image Classification — Level 3: Transfer Learning
# ============================================================
# FINAL YEAR PROJECT — MCA
# Student: Ashwin B
# Roll No: AA.SC.P2MCA25010034
#
# This script implements Transfer Learning using VGG16 pre-trained
# on ImageNet (14 million images). Instead of learning from scratch,
# we leverage a model that already understands visual features.
#
# WHAT IS TRANSFER LEARNING?
# ─────────────────────────
# Think of it like this:
#   - Level 1-2: Teaching a baby to recognize cats vs dogs (from zero)
#   - Level 3:   Hiring an expert photographer and saying
#                "You already know what eyes, ears, fur look like.
#                 Just tell me if it's a cat or a dog."
#
# VGG16 was trained on 14 million images across 1000 categories.
# It already knows how to detect:
#   Layer 1-2:  Edges, lines, colors
#   Layer 3-5:  Textures, patterns
#   Layer 6-10: Shapes (ears, eyes, noses)
#   Layer 11-13: Complex objects (faces, bodies)
#
# We FREEZE these layers (keep their knowledge) and only train
# a new "head" that decides: cat or dog?
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# --- Core TensorFlow/Keras imports ---
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Transfer Learning: Pre-trained models ---
# VGG16: A 16-layer deep CNN trained on ImageNet
# "imagenet" weights = pre-learned feature detectors from 14 million images
from tensorflow.keras.applications import VGG16

# --- Evaluation metrics ---
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = 224                 # VGG16 requires exactly 224x224 input
BATCH_SIZE = 32                # Batch size for training
EPOCHS = 30                    # Max epochs (early stopping will likely stop earlier)
LEARNING_RATE = 0.0001         # Initial learning rate
FINE_TUNE_LEARNING_RATE = 0.00001  # Lower LR for fine-tuning phase


# ============================================================
# STEP 2: DATA PREPROCESSING & AUGMENTATION
# ============================================================
# For transfer learning, we use the SAME preprocessing that VGG16
# was originally trained with. This is critical for good performance.
#
# DATA AUGMENTATION (for training only):
# Since we want a fool-proof model, we use smart augmentation:
#   - Rotation, shifting, zooming = handle different photo angles
#   - Horizontal flip = cats/dogs look similar flipped
#   - Brightness range = handle different lighting conditions
#   - Shear = handle perspective distortion
#   - validation_split = automatically split 15% for validation

print("=" * 60)
print("  CAT vs DOG CLASSIFIER — Level 3: Transfer Learning")
print("  Using VGG16 pre-trained on ImageNet (14M images)")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,       # Re-enabled: with large dataset, augmentation helps
    zoom_range=0.15,
    shear_range=0.1,            # Slight perspective distortion
    brightness_range=[0.8, 1.2], # Handle different lighting
    validation_split=0.15       # 15% of training data used for validation
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# --- Load training images (85% of dataset/train) ---
print("\nLoading training images...")
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",          # Use 85% for training
    shuffle=True
)

# --- Load validation images (15% of dataset/train) ---
print("Loading validation images...")
val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",        # Use 15% for validation
    shuffle=False
)

# --- Load test images (completely separate, never seen during training) ---
print("Loading test images...")
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False               # Don't shuffle test data (for evaluation)
)

print(f"\nClass labels: {train_data.class_indices}")
print(f"Training samples:   {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Test samples:       {test_data.samples}")


# ============================================================
# STEP 3: BUILD THE TRANSFER LEARNING MODEL
# ============================================================
# Architecture:
#   [VGG16 Base — FROZEN] → [GlobalAvgPool] → [Dense 512] → [Dense 256] → [Output]
#
# WHY VGG16?
# - Simple, well-understood architecture
# - Excellent for image classification
# - 16 layers deep with 138M parameters (all pre-trained!)
# - Your proposal specifically mentions VGG16
#
# WHAT IS "FREEZING"?
# - Frozen layers = their weights DON'T change during training
# - We keep VGG16's knowledge intact
# - Only train the new layers we add on top

print("\n" + "=" * 60)
print("  BUILDING VGG16 TRANSFER LEARNING MODEL")
print("=" * 60)

# Load VGG16 WITHOUT the top classification layers
# include_top=False removes the final Dense layers (meant for 1000 ImageNet classes)
# We'll add our own Dense layers for cat vs dog (2 classes)
print("\nDownloading VGG16 pre-trained weights (first time only)...")
base_model = VGG16(
    weights="imagenet",         # Use pre-trained ImageNet weights
    include_top=False,          # Remove original classification head
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FREEZE all VGG16 layers — keep their learned features
# This is the key trick: we don't retrain 138M parameters!
for layer in base_model.layers:
    layer.trainable = False

print(f"VGG16 base loaded: {len(base_model.layers)} layers, ALL FROZEN")
print(f"VGG16 parameters: {base_model.count_params():,} (none trainable)")

# --- Add our custom classification head ---
# GlobalAveragePooling2D: Better than Flatten for transfer learning
#   - Flatten: 7x7x512 = 25,088 features → too many, causes overfitting
#   - GAP: Averages each 7x7 feature map into 1 number → 512 features (cleaner!)

x = base_model.output
x = GlobalAveragePooling2D()(x)     # 7x7x512 → 512 features

x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Output: 1 neuron, sigmoid activation (0=cat, 1=dog)
predictions = Dense(1, activation="sigmoid")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Count parameters
total_params = model.count_params()
trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
frozen_params = total_params - trainable_params

print(f"\nFINAL MODEL ARCHITECTURE:")
print(f"  Total parameters:     {total_params:>12,}")
print(f"  Frozen (VGG16):       {frozen_params:>12,}  (pre-trained, kept intact)")
print(f"  Trainable (new head): {trainable_params:>12,}  (we train only these)")
print(f"  % trainable:          {trainable_params/total_params*100:>11.2f}%")

model.summary()


# ============================================================
# STEP 4: COMPILE & TRAIN — PHASE 1 (Feature Extraction)
# ============================================================
# Phase 1: Train ONLY the new head layers (VGG16 stays frozen)
# This teaches our new layers to use VGG16's features for cat/dog

print("\n" + "=" * 60)
print("  PHASE 1: FEATURE EXTRACTION (VGG16 frozen)")
print("  Training only the classification head...")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks for smart training
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Save the best model during training
checkpoint = ModelCheckpoint(
    "cat_dog_model_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

print(f"\nTraining on {train_data.samples} images...")
print(f"Validating on {val_data.samples} images...\n")

history_phase1 = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)


# ============================================================
# STEP 5: FINE-TUNING — PHASE 2 (Unfreeze top VGG16 layers)
# ============================================================
# Now that our head is trained, we "unfreeze" the last few VGG16 layers
# and train them with a VERY low learning rate.
#
# WHY FINE-TUNE?
# - The top VGG16 layers detect high-level features (faces, body shapes)
# - By fine-tuning, we adapt these to be SPECIFIC to cats vs dogs
# - Use very low learning rate to avoid destroying pre-trained knowledge

print("\n" + "=" * 60)
print("  PHASE 2: FINE-TUNING (Unfreezing top VGG16 layers)")
print("  Adapting high-level features for cat vs dog...")
print("=" * 60)

# Unfreeze the last 4 layers of VGG16
# VGG16 has 19 layers. We unfreeze layers 15-19 (the last conv block)
for layer in base_model.layers[-4:]:
    layer.trainable = True

trainable_after = sum([np.prod(w.shape) for w in model.trainable_weights])
print(f"\nUnfrozen layers: {[l.name for l in base_model.layers[-4:]]}")
print(f"Trainable parameters now: {trainable_after:,}")

# Re-compile with LOWER learning rate (critical for fine-tuning!)
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Reset callbacks for Phase 2
early_stopping_ft = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_ft = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

checkpoint_ft = ModelCheckpoint(
    "cat_dog_model_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history_phase2 = model.fit(
    train_data,
    epochs=20,                  # Fine-tuning needs fewer epochs
    validation_data=val_data,
    callbacks=[early_stopping_ft, reduce_lr_ft, checkpoint_ft],
    verbose=1
)


# ============================================================
# STEP 6: EVALUATE ON TEST SET
# ============================================================
# Final honest evaluation on images the model has NEVER seen

print("\n" + "=" * 60)
print("  FINAL EVALUATION ON TEST SET")
print("=" * 60)

# Load the best saved model
best_model = load_model("cat_dog_model_best.h5")

test_loss, test_accuracy = best_model.evaluate(test_data, verbose=1)
print(f"\n{'='*40}")
print(f"  TEST ACCURACY: {test_accuracy*100:.2f}%")
print(f"  TEST LOSS:     {test_loss:.4f}")
print(f"{'='*40}")

# Detailed classification report
test_data.reset()
predictions = best_model.predict(test_data, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_data.classes

class_names = list(test_data.class_indices.keys())
print(f"\nDETAILED CLASSIFICATION REPORT:")
print("-" * 40)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print(f"CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"                Cat    Dog")
print(f"Actual   Cat   [{cm[0,0]:4d}  {cm[0,1]:4d}]")
print(f"         Dog   [{cm[1,0]:4d}  {cm[1,1]:4d}]")


# ============================================================
# STEP 7: SAVE FINAL MODEL & PLOT RESULTS
# ============================================================

# Save final model
model.save("cat_dog_model.h5")
print(f"\nFinal model saved as 'cat_dog_model.h5'")
print(f"Best model saved as 'cat_dog_model_best.h5'")

# --- Combined training plots (Phase 1 + Phase 2) ---
acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
loss = history_phase1.history["loss"] + history_phase2.history["loss"]
val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]

phase1_epochs = len(history_phase1.history["accuracy"])

plt.figure(figsize=(14, 5))

# --- Accuracy Plot ---
plt.subplot(1, 2, 1)
plt.plot(acc, label="Training Accuracy", linewidth=2)
plt.plot(val_acc, label="Validation Accuracy", linewidth=2)
plt.axvline(x=phase1_epochs, color="red", linestyle="--", label="Fine-tuning starts")
plt.title("Model Accuracy (VGG16 Transfer Learning)", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Loss Plot ---
plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss", linewidth=2)
plt.plot(val_loss, label="Validation Loss", linewidth=2)
plt.axvline(x=phase1_epochs, color="red", linestyle="--", label="Fine-tuning starts")
plt.title("Model Loss (VGG16 Transfer Learning)", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("Training graphs saved as 'training_results.png'")
plt.show()

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print(f"  Final Test Accuracy: {test_accuracy*100:.2f}%")
print("  Run 'python test.py <image_path>' to make predictions")
print("=" * 60)
