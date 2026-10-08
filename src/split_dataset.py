from pathlib import Path
import csv
import random
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "splits.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Train, validation, and test ratios must sum to 1.0")


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

def create_patient_split(rows):
    """
    Split patients rather than individual MRI slices.

    This prevents images belonging to the same patient from
    appearing in different dataset splits.
    """

    patient_to_rows = {}

    for row in rows:
        patient_id = row["patient_id"]

        patient_to_rows.setdefault(patient_id, []).append(row)

    patients = list(patient_to_rows.keys())

    random.seed(RANDOM_SEED)
    random.shuffle(patients)

    total_patients = len(patients)

    train_count = round(total_patients * TRAIN_RATIO)
    val_count = round(total_patients * VAL_RATIO)

    # Ensure the three groups contain every patient exactly once.
    train_patients = patients[:train_count]

    val_patients = patients[
        train_count:train_count + val_count
    ]

    test_patients = patients[
        train_count + val_count:
    ]

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    if train_set & val_set:
        raise RuntimeError("Patient overlap between train and validation.")

    if train_set & test_set:
        raise RuntimeError("Patient overlap between train and test.")

    if val_set & test_set:
        raise RuntimeError("Patient overlap between validation and test.")

    split_rows = []

    for patient_id, patient_rows in patient_to_rows.items():

        if patient_id in train_set:
            split = "train"

        elif patient_id in val_set:
            split = "val"

        elif patient_id in test_set:
            split = "test"

        else:
            raise RuntimeError(
                f"Patient {patient_id} was not assigned to a split."
            )

        for row in patient_rows:
            new_row = dict(row)
            new_row["split"] = split
            split_rows.append(new_row)

    return split_rows


# ============================================================
# VALIDATION
# ============================================================

def validate_split(rows):
    print()
    print("=" * 60)
    print("SPLIT VALIDATION")
    print("=" * 60)

    split_patient_ids = {}

    for split_name in ["train", "val", "test"]:

        patients = {
            row["patient_id"]
            for row in rows
            if row["split"] == split_name
        }

        split_patient_ids[split_name] = patients

    train_patients = split_patient_ids["train"]
    val_patients = split_patient_ids["val"]
    test_patients = split_patient_ids["test"]

    # --------------------------------------------------------
    # Patient overlap check
    # --------------------------------------------------------

    overlap_train_val = train_patients & val_patients
    overlap_train_test = train_patients & test_patients
    overlap_val_test = val_patients & test_patients

    if overlap_train_val:
        raise RuntimeError("Train/validation patient leakage detected.")

    if overlap_train_test:
        raise RuntimeError("Train/test patient leakage detected.")

    if overlap_val_test:
        raise RuntimeError("Validation/test patient leakage detected.")

    # --------------------------------------------------------
    # Print patient counts
    # --------------------------------------------------------

    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    print()
    print("Patient overlap check: PASSED")

    # --------------------------------------------------------
    # Image counts
    # --------------------------------------------------------

    print()
    print("Image distribution:")

    for split_name in ["train", "val", "test"]:

        split_rows = [
            row for row in rows
            if row["split"] == split_name
        ]

        print(
            f"  {split_name}: {len(split_rows)} images"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("Class distribution:")

    for split_name in ["train", "val", "test"]:

        split_rows = [
            row for row in rows
            if row["split"] == split_name
        ]

        counts = Counter(
            row["class_name"]
            for row in split_rows
        )

        print()
        print(f"{split_name}:")

        for class_name in [
            "meningioma",
            "glioma",
            "pituitary",
        ]:
            print(
                f"  {class_name}: {counts[class_name]}"
            )


# ============================================================
# SAVE SPLIT FILE
# ============================================================

def save_split(rows):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "file_path",
        "label",
        "class_name",
        "patient_id",
        "split",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("XMedAI PATIENT-LEVEL DATASET SPLIT")
    print("=" * 60)

    rows = load_metadata()

    print(f"Loaded images: {len(rows)}")

    unique_patients = {
        row["patient_id"]
        for row in rows
    }

    print(f"Loaded patients: {len(unique_patients)}")

    split_rows = create_patient_split(rows)

    validate_split(split_rows)

    save_split(split_rows)

    print()
    print("=" * 60)
    print("SPLIT FILE CREATED")
    print("=" * 60)

    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Random seed:", RANDOM_SEED)
    print(
        f"Ratios: "
        f"{TRAIN_RATIO:.0%} train / "
        f"{VAL_RATIO:.0%} validation / "
        f"{TEST_RATIO:.0%} test"
    )


if __name__ == "__main__":
    main()