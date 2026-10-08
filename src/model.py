import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = [
    "meningioma",
    "glioma",
    "pituitary",
]


# ============================================================
# CREATE RESNET18 MODEL
# ============================================================

def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """
    Create a ResNet18 model for brain tumor classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.

    Returns:
        PyTorch ResNet18 model.
    """

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(
        weights=weights
    )

    # Replace the original ImageNet classifier.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("XMedAI RESNET18 MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)

    print()
    print("Model: ResNet18")
    print("Number of classes:", NUM_CLASSES)
    print("Classes:", CLASS_NAMES)

    # Dummy batch matching dataset output.
    dummy_input = torch.randn(
        4,
        3,
        224,
        224,
    ).to(device)

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print()
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    expected_shape = (
        4,
        NUM_CLASSES,
    )

    actual_shape = tuple(
        output.shape
    )

    if actual_shape != expected_shape:

        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{actual_shape}. "
            f"Expected: {expected_shape}"
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print()
    print("=" * 60)
    print("MODEL TEST PASSED")
    print("=" * 60)