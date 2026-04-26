import matplotlib.pyplot as plt
import os

# ==============
# OUTPUT FOLDER
# ==============
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# ==============
# MODEL RESULTS
# ==============
models = [
    "Custom CNN",
    "ResNet FE",
    "ResNet FT",
    "EffNet FE",
    "EffNet FT"
]

accuracy = [0.449, 0.705, 0.793, 0.687, 0.706]

# ===================
# ACCURACY BAR CHART
# ===================
plt.figure(figsize=(10,5))
bars = plt.bar(models, accuracy)

# Add values on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.3f}', ha='center', va='bottom')

plt.title("Model Comparison (Validation Accuracy)")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.ylim(0,1)

plt.savefig(os.path.join(output_folder, "model_comparison.png"))
plt.show()

# =======================
# FINE-TUNING IMPROVEMENT
# =======================
improvement = [
    0,
    0,
    0.793 - 0.705,  # ResNet improvement
    0,
    0.706 - 0.687   # EfficientNet improvement
]

plt.figure(figsize=(10,5))
bars = plt.bar(models, improvement)

# Add values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.3f}', ha='center', va='bottom')

plt.title("Fine-Tuning Improvement")
plt.ylabel("Accuracy Gain")
plt.xticks(rotation=20)

plt.savefig(os.path.join(output_folder, "fine_tuning_improvement.png"))
plt.show()

# ==============
# PRINT SUMMARY
# ==============
best_model_index = accuracy.index(max(accuracy))

print("\n=== FINAL COMPARISON ===")
print(f"Best Model: {models[best_model_index]}")
print(f"Best Accuracy: {max(accuracy):.4f}")

