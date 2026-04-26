import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

dataset_path = "Dataset"
img_size = (224, 224)
batch_size = 32

# ================
# LOAD TRAIN DATA
# ================
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# ======================
# LOAD VALIDATION DATA
# ======================
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False   
)

# ============
# CLASS NAMES
# ============
class_names = train_dataset.class_names
print("Classes:", class_names)

# ==================================
# EXTRACT LABELS (for class weights)
# ==================================
labels = np.array([y.numpy() for x, y in train_dataset.unbatch()])

# ==============
# CLASS WEIGHTS
# ==============
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weights)

# =========================
# PERFORMANCE OPTIMISATION
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)