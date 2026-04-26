import tensorflow as tf
import matplotlib.pyplot as plt
import os

# =====================
# IMPORT PREPROCESSING
# =====================
from DataPreProcessing import train_dataset, val_dataset, class_weights

num_classes = 8

# ==============
# OUTPUT FOLDER
# ==============
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

# =================
# BUILD CUSTOM CNN
# =================
print("\n=== CUSTOM CNN TRAINING (SCRATCH) ===")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    
    data_augmentation,
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# ==============
# COMPILE MODEL
# ==============
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============
# TRAIN MODEL
# ============
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    class_weight=class_weights
)

# ===========
# SAVE MODEL
# ===========
model.save(os.path.join(output_folder, "custom_cnn.keras"))

# ======================
# PLOT ACCURACY
# ======================
plt.figure()
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Custom CNN Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_folder, "custom_cnn_accuracy.png"))
plt.show()

# ==========
# PLOT LOSS
# ==========
plt.figure()
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Custom CNN Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_folder, "custom_cnn_loss.png"))
plt.show()

# ==============
# FINAL RESULT
# ==============
final_acc = max(history.history['val_accuracy'])

print("\n=== FINAL RESULT ===")
print(f"Custom CNN Accuracy: {final_acc:.4f}")

