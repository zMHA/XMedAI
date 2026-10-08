from pathlib import Path
import csv

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "splits.csv"

IMAGE_SIZE = 224

CLASS_NAMES = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


# ImageNet normalization values.
# These are used because we will use a pretrained ResNet18.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms(split):
    """
    Create the preprocessing/augmentation pipeline.

    Train:
        - Resize
        - Mild rotation
        - Mild random resized crop
        - Horizontal flip
        - Grayscale -> 3 channels
        - ToTensor
        - ImageNet normalization

    Validation/Test:
        - Resize
        - Grayscale -> 3 channels
        - ToTensor
        - ImageNet normalization
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split must be 'train', 'val', or 'test'"
        )

    if split == "train":

        return transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomRotation(
                degrees=10
            ),

            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])

    else:

        return transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])


# ============================================================
# BRAIN TUMOR DATASET
# ============================================================

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for the Jun Cheng Brain Tumor Dataset.

    Each sample is loaded directly from its original MATLAB
    v7.3 (.mat) file using h5py.

    Labels returned by the dataset are zero-based:

        0 -> meningioma
        1 -> glioma
        2 -> pituitary

    Returned sample:

        image
        label
        patient_id
        file_path
        class_name
    """

    def __init__(
        self,
        split,
        transform=None,
    ):
        """
        Parameters
        ----------
        split : str
            One of: "train", "val", "test"

        transform : callable, optional
            Torchvision transformation pipeline.

            If None, the default pipeline for the selected
            split is automatically used.
        """

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        self.split = split

        # Use the default preprocessing pipeline if the caller
        # does not provide a custom transform.
        if transform is None:
            transform = get_transforms(split)

        self.transform = transform

        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split: {split}"
            )


    # ========================================================
    # LOAD METADATA
    # ========================================================

    def _load_samples(self):

        samples = []

        with open(
            SPLIT_FILE,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["split"] != self.split:
                    continue

                samples.append(
                    {
                        "file_path": row["file_path"],
                        "label": int(row["label"]),
                        "class_name": row["class_name"],
                        "patient_id": row["patient_id"],
                    }
                )

        return samples


    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):
        return len(self.samples)


    # ========================================================
    # LOAD ONE MRI
    # ========================================================

    def _load_image(self, file_path):

        absolute_path = PROJECT_ROOT / file_path

        with h5py.File(
            absolute_path,
            "r"
        ) as file:

            image = np.array(
                file["cjdata"]["image"]
            )

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Min-max intensity normalization
        # ----------------------------------------------------

        min_value = image.min()
        max_value = image.max()

        if max_value > min_value:

            image = (
                (image - min_value)
                / (max_value - min_value)
                * 255.0
            )

        else:

            image = np.zeros_like(
                image
            )

        # ----------------------------------------------------
        # Convert to uint8 for PIL
        # ----------------------------------------------------

        image = image.astype(
            np.uint8
        )

        return Image.fromarray(
            image,
            mode="L"
        )


    # ========================================================
    # GET ONE SAMPLE
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self._load_image(
            sample["file_path"]
        )

        # Original dataset labels:
        #
        # 1 = meningioma
        # 2 = glioma
        # 3 = pituitary
        #
        # Convert to zero-based labels for PyTorch:
        #
        # 0 = meningioma
        # 1 = glioma
        # 2 = pituitary

        label = sample["label"] - 1

        if self.transform is not None:

            image = self.transform(
                image
            )

        return {
            "image": image,

            "label": torch.tensor(
                label,
                dtype=torch.long
            ),

            "patient_id": sample[
                "patient_id"
            ],

            "file_path": sample[
                "file_path"
            ],

            "class_name": sample[
                "class_name"
            ],
        }