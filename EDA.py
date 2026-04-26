# =================
# IMPORT LIBRARIES
# =================
import os                          # for folder/file handling
from PIL import Image              # for opening images
import matplotlib.pyplot as plt    # for plotting

# ===============================
# DATASET PATH AND OUTPUT FOLDER
# ===============================
dataset_path = "Dataset"   
output_folder = "Results"

# ===================
# STORE CLASS COUNTS
# ===================
class_names = []     # store class labels
image_counts = []    # store number of images per class

# ================================
# CREATE FIGURE FOR SAMPLE IMAGES
# ================================
plt.figure(figsize=(10, 8))
i = 1   # subplot index

# =====================
# LOOP THROUGH DATASET
# =====================
for label in os.listdir(dataset_path):

    label_path = os.path.join(dataset_path, label)

    # check it's a folder
    if os.path.isdir(label_path):

        # count images in this class
        images = os.listdir(label_path)
        image_count = len(images)

        # store for graph later
        class_names.append(label)
        image_counts.append(image_count)

        # print class info
        print(f"{label}: {image_count} images")

        # get first image as sample
        sample_image_name = images[0]
        sample_image_path = os.path.join(label_path, sample_image_name)

        # open image
        img = Image.open(sample_image_path)

        # print image size
        print(f"Sample image size for {label}: {img.size}\n")

        # display image
        plt.subplot(3, 3, i)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

        i += 1

# ===================
# SHOW SAMPLE IMAGES
# ===================
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "SAMPLE_IMAGES.png"))
plt.show()

# =========================
# CLASS DISTRIBUTION GRAPH
# =========================
plt.figure()

plt.bar(class_names, image_counts)
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "CLASS_DISTRIBUTION_GRAPH.png"))
plt.show()