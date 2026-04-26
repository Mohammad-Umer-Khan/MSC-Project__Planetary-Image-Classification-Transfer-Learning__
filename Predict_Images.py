import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ==========
# LOAD MODEL
# ==========
model = tf.keras.models.load_model("Results/resnet50_finetuned.keras")

# ============
# CLASS NAMES
# ============
class_names = [
    'asteroid', 'black hole', 'comet', 'constellation',
    'galaxy', 'nebula', 'planet', 'star'
]

# =========
# SETTINGS
# =========
img_size = (224, 224)
test_folder = "TestImages"

# ===============
# PREDICT IMAGES
# ===============
print("\n=== IMAGE PREDICTIONS ===\n")

for img_name in os.listdir(test_folder):

    img_path = os.path.join(test_folder, img_name)

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # IMPORTANT: ResNet preprocessing
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Print result
        print(f"{img_name} → {predicted_class} ({confidence:.2f})")

        # Display image
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.2f})")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error processing {img_name}: {e}")