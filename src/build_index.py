from pathlib import Path
import csv
import h5py


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "metadata.csv"


# Dataset labels from the official README:
# 1 = meningioma
# 2 = glioma
# 3 = pituitary tumor

CLASS_NAMES = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_scalar(dataset):
    """Extract a scalar value from a small HDF5 dataset."""
    value = dataset[()]
    return value.flatten()[0]


def extract_patient_id(dataset):
    """Extract the patient ID stored as a 6x1 uint16 array."""
    value = dataset[()]
    return "".join(str(int(x)) for x in value.flatten())


def read_mat_metadata(mat_path):
    """Read label and patient ID from one .mat file."""
    with h5py.File(mat_path, "r") as file:
        cjdata = file["cjdata"]

        label = int(extract_scalar(cjdata["label"]))
        patient_id = extract_patient_id(cjdata["PID"])

    return label, patient_id


# ============================================================
# MAIN INDEX BUILDING FUNCTION
# ============================================================

def build_index():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(
        path
        for path in RAW_DIR.rglob("*.mat")
        if path.name.lower() != "cvind.mat"
    )

    print(f"Found {len(mat_files)} .mat files.")

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found inside: {RAW_DIR}"
        )

    rows = []

    for index, mat_path in enumerate(mat_files, start=1):

        label, patient_id = read_mat_metadata(mat_path)

        if label not in CLASS_NAMES:
            raise ValueError(
                f"Unknown label {label} found in {mat_path}"
            )

        relative_path = mat_path.relative_to(PROJECT_ROOT)

        rows.append(
            {
                "file_path": str(relative_path).replace("\\", "/"),
                "label": label,
                "class_name": CLASS_NAMES[label],
                "patient_id": patient_id,
            }
        )

        if index % 250 == 0 or index == len(mat_files):
            print(f"Processed {index}/{len(mat_files)} files.")


    # ========================================================
    # WRITE CSV
    # ========================================================

    fieldnames = [
        "file_path",
        "label",
        "class_name",
        "patient_id",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


    # ========================================================
    # SUMMARY
    # ========================================================

    label_counts = {}
    patient_ids = set()

    for row in rows:
        label = int(row["label"])

        label_counts[label] = label_counts.get(label, 0) + 1
        patient_ids.add(row["patient_id"])

    print()
    print("=" * 60)
    print("DATASET INDEX CREATED")
    print("=" * 60)

    print(f"Total images: {len(rows)}")
    print(f"Unique patients: {len(patient_ids)}")

    print()
    print("Class distribution:")

    for label, class_name in CLASS_NAMES.items():
        count = label_counts.get(label, 0)
        print(f"  {label} - {class_name}: {count}")

    print()
    print(f"Metadata file: {OUTPUT_FILE}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_index()