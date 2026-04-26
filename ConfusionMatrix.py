import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========
# LOAD DATA
# ==========
from DataPreProcessing import val_dataset

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("Results/resnet50_finetuned.keras",compile=False)

# ============
# CLASS NAMES
# ============
class_names = [
    'asteroid', 'black hole', 'comet', 'constellation',
    'galaxy', 'nebula', 'planet', 'star'
]

# ================
# GET PREDICTIONS
# ================
y_true = []
y_pred = []

for images, labels in val_dataset:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =================
# CONFUSION MATRIX
# =================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(xticks_rotation=45)
plt.title("ResNet50 Confusion Matrix")
plt.savefig("Results/resnet_confusion_matrix.png")
plt.show()