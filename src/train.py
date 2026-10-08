from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility
RANDOM_SEED = 42


# Training configuration
BATCH_SIZE = 16
NUM_EPOCHS = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0


# Model configuration
NUM_CLASSES = 3


CLASS_NAMES = [
    "meningioma",
    "glioma",
    "pituitary",
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_dataloaders():

    train_dataset = BrainTumorDataset(
        split="train"
    )

    val_dataset = BrainTumorDataset(
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# ============================================================
# FREEZE RESNET18 BACKBONE
# ============================================================

def freeze_backbone(model):

    for parameter in model.parameters():

        parameter.requires_grad = False

    # Only the final classifier is trainable.
    for parameter in model.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("XMedAI - RESNET18 TRAINING")
    print("=" * 60)

    set_seed(RANDOM_SEED)

    device = get_device()

    print()
    print("Device:", device)

    print()
    print("Configuration")
    print("-" * 60)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Random seed:", RANDOM_SEED)
    print("Frozen backbone: YES")
    print("Trainable layer: model.fc")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = create_dataloaders()

    print()
    print("Dataset")
    print("-" * 60)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    freeze_backbone(model)

    model = model.to(device)

    # --------------------------------------------------------
    # CHECK TRAINABLE PARAMETERS
    # --------------------------------------------------------

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Model")
    print("-" * 60)
    print("Architecture: ResNet18")
    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    best_model_path = (
        CHECKPOINT_DIR
        / "resnet18_stage1_best.pth"
    )

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print()
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "class_names": CLASS_NAMES,
                    "random_seed": RANDOM_SEED,
                },
                best_model_path,
            )

            print(
                "✓ Best model saved."
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy * 100:.2f}%",
    )

    print(
        "Best model:",
        best_model_path,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()