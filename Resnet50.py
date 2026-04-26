import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================
# IMPORT PREPROCESSING
# =====================
from DataPreProcessing import train_dataset, val_dataset, class_weights, class_names

num_classes = 8

# =====================
# CREATE OUTPUT FOLDER
# =====================
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# ==================
# DATA AUGMENTATION
# ==================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ======================
# LOAD PRETRAINED MODEL
# ======================
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# ===================
# FEATURE EXTRACTION
# ===================
print("\n=== FEATURE EXTRACTION ===")

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fe = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights
)

model.save(os.path.join(output_folder, "resnet50_feature_extraction.keras"))

# ============
# FINE TUNING
# ============
print("\n=== FINE TUNING ===")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights
)

model.save(os.path.join(output_folder, "resnet50_finetuned.keras"))

# ==============
# PLOT ACCURACY
# ==============
plt.figure(figsize=(10,5))
plt.plot(history_fe.history['val_accuracy'], label='Feature Extraction')
plt.plot(history_ft.history['val_accuracy'], label='Fine Tuning')
plt.title('ResNet50 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_folder, "resnet50_accuracy.png"))
plt.show()

# ==========
# PLOT LOSS
# ==========
plt.figure(figsize=(10,5))
plt.plot(history_fe.history['val_loss'], label='Feature Extraction')
plt.plot(history_ft.history['val_loss'], label='Fine Tuning')
plt.title('ResNet50 Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_folder, "resnet50_loss.png"))
plt.show()

# ==============
# FINAL RESULTS
# ==============
fe_acc = max(history_fe.history['val_accuracy'])
ft_acc = max(history_ft.history['val_accuracy'])

print("\n=== FINAL RESULTS ===")
print(f"Feature Extraction Accuracy: {fe_acc:.4f}")
print(f"Fine Tuning Accuracy: {ft_acc:.4f}")

