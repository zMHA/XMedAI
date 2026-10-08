import torch
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset


# ============================================================
# TEST EACH DATASET SPLIT
# ============================================================

for split in ["train", "val", "test"]:

    print()
    print("=" * 60)
    print(f"Testing split: {split}")
    print("=" * 60)

    dataset = BrainTumorDataset(
        split=split
    )

    print(
        "Dataset length:",
        len(dataset)
    )

    sample = dataset[0]

    print(
        "Image shape:",
        sample["image"].shape
    )

    print(
        "Image dtype:",
        sample["image"].dtype
    )

    print(
        "Image min:",
        float(sample["image"].min())
    )

    print(
        "Image max:",
        float(sample["image"].max())
    )

    print(
        "Label:",
        sample["label"].item()
    )

    print(
        "Class:",
        sample["class_name"]
    )

    print(
        "Patient ID:",
        sample["patient_id"]
    )

    print(
        "File:",
        sample["file_path"]
    )


# ============================================================
# CHECK BATCH
# ============================================================

dataset = BrainTumorDataset(
    split="train"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

batch = next(iter(loader))


print()
print("=" * 60)
print("BATCH TEST")
print("=" * 60)

print(
    "Images:",
    batch["image"].shape
)

print(
    "Labels:",
    batch["label"].shape
)

print(
    "Labels:",
    batch["label"].tolist()
)

print(
    "Patient IDs:",
    batch["patient_id"]
)


# ============================================================
# FINAL VALIDATION
# ============================================================

expected_shape = (
    4,
    3,
    224,
    224
)

actual_shape = tuple(
    batch["image"].shape
)

if actual_shape != expected_shape:

    raise RuntimeError(
        f"Unexpected batch shape: "
        f"{actual_shape}. "
        f"Expected: {expected_shape}"
    )


print()
print("=" * 60)
print("DATASET TEST PASSED")
print("=" * 60)