import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

# --- Step 1: Load pretrained ResNet50 (no top layer since we'll just see features) ---
model = ResNet50(weights='imagenet', include_top=True)  # include_top=True for full classifier

# --- Step 2: Load dataset from folder structure ---
dataset_dir = "Dataset"  # top folder containing subfolders like asteroid, galaxy, etc.

# Load a small subset for testing
batch_size = 4
img_height = 224
img_width = 224

test_dataset = image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True
)

class_names = test_dataset.class_names
print("Detected classes:", class_names)

# --- Step 3: Preprocess and run predictions on one batch ---
for images, labels in test_dataset.take(1):  # take just 1 batch for testing
    images_preprocessed = preprocess_input(images)  # preprocess for ResNet50
    preds = model.predict(images_preprocessed)
    
    # decode_predictions works for ImageNet classes
    from tensorflow.keras.applications.resnet50 import decode_predictions
    decoded = decode_predictions(preds, top=3)
    
    for i, d in enumerate(decoded):
        print(f"\nImage {i} predictions (ImageNet classes):")
        for _, class_name, prob in d:
            print(f"{class_name}: {prob:.4f}")