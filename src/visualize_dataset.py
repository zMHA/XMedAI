from pathlib import Path
import random

import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import BrainTumorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "dataset_samples.png"


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

dataset = BrainTumorDataset(
    split="test",
    transform=transform,
)


# ============================================================
# SELECT SAMPLE IMAGES
# ============================================================

random_generator = random.Random(42)

sample_count = 6
sample_indices = random_generator.sample(
    range(len(dataset)),
    sample_count,
)


# ============================================================
# DISPLAY IMAGES
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, index in zip(axes, sample_indices):

    sample = dataset[index]

    image = sample["image"]
    label = sample["label"].item()
    class_name = sample["class_name"]
    patient_id = sample["patient_id"]

    # Convert [1, H, W] -> [H, W]
    image = image.squeeze(0)

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"{class_name}\nPID: {patient_id}",
        fontsize=10,
    )
    ax.axis("off")


# ============================================================
# SAVE RESULT
# ============================================================

fig.suptitle(
    "XMedAI Brain Tumor MRI Dataset Samples",
    fontsize=16,
)

plt.tight_layout()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight",
)

print("=" * 60)
print("MRI VISUALIZATION")
print("=" * 60)

print(f"Dataset: test")
print(f"Total test images: {len(dataset)}")
print(f"Displayed samples: {sample_count}")
print(f"Saved visualization: {OUTPUT_FILE}")

print("=" * 60)

plt.show()