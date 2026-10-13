# XMedAI

### Explainable and Reliable AI for Medical Image Analysis

XMedAI is a medical AI research project focused on deep learning for brain tumor MRI classification. The project aims to build a reproducible pipeline for medical image classification, with an emphasis on patient-level data splitting, reliable model evaluation, and future explainability analysis.

The initial implementation uses a Brain Tumor MRI dataset and an ImageNet-pretrained ResNet18 model to establish a baseline for classifying three types of brain tumors.

## Research Objectives

* Develop a reproducible pipeline for brain MRI classification.
* Evaluate transfer learning using a pretrained convolutional neural network.
* Reduce data leakage by separating patients across training, validation, and test sets.
* Establish a baseline for future experiments with explainable AI (XAI).
* Investigate model reliability and generalization on unseen patients.

## Dataset

The project uses a Brain Tumor MRI dataset containing **3,064 MATLAB (`.mat`) image slices** across three tumor classes.

| Class      | Description                                             |
| ---------- | ------------------------------------------------------- |
| Meningioma | Brain tumors originating from the meninges              |
| Glioma     | Tumors arising from glial cells or related brain tissue |
| Pituitary  | Tumors associated with the pituitary gland              |

### Patient-Level Data Split

The dataset is split at the patient level to reduce the risk of data leakage between training, validation, and test sets.

| Split      |    Images | Patients |
| ---------- | --------: | -------: |
| Training   |     2,145 |      214 |
| Validation |       481 |       35 |
| Test       |       438 |       35 |
| **Total**  | **3,064** |  **284** |

**Important:** The test set is kept separate from model training and validation. Final test performance should be reported only after the evaluation is conducted.

## Model Architecture

The initial baseline uses **ResNet18 with ImageNet-pretrained weights**.

### Preprocessing

* Load MRI slices from MATLAB files.
* Convert grayscale images into three-channel images.
* Resize images to `224 × 224` pixels.
* Apply normalization appropriate for the pretrained model.

### Transfer Learning

The initial training stage freezes the ResNet18 backbone and trains the final classification layer for the three tumor classes.

| Configuration                   | Value                       |
| ------------------------------- | --------------------------- |
| Architecture                    | ResNet18                    |
| Initialization                  | ImageNet-pretrained weights |
| Input shape                     | `3 × 224 × 224`             |
| Number of classes               | 3                           |
| Total parameters                | 11,178,051                  |
| Trainable parameters in Stage 1 | 1,539                       |
| Initial training duration       | 10 epochs                   |

## Initial Results

The first training stage achieved a **best validation accuracy of 87.32% at epoch 9**.

| Metric                   |                                   Result |
| ------------------------ | ---------------------------------------: |
| Best validation accuracy |                                   87.32% |
| Best epoch               |                                        9 |
| Training stage           | Frozen backbone, trained classifier head |

These results establish an initial baseline, not a final measure of clinical performance. Test-set evaluation, per-class precision and recall, F1-score, confusion matrix analysis, and robustness assessment are needed for a more complete evaluation.

## Explainable AI Roadmap

A central research direction for XMedAI is to investigate how medical image models arrive at their predictions.

Planned research directions include:

* **Grad-CAM:** Visualize image regions that influence classification decisions.
* **Model comparison:** Evaluate alternative architectures and fine-tuning strategies.
* **Class-wise evaluation:** Examine performance differences across tumor categories.
* **Reliability analysis:** Investigate generalization, prediction errors, and potential data leakage.
* **Robustness evaluation:** Study how image transformations and quality variations affect predictions.

These are planned extensions unless explicitly documented as completed in the repository.

## Project Structure

```text
XMedAI/
├── data/
│   └── processed/
│       ├── metadata.csv
│       └── splits.csv
├── models/
├── src/
│   ├── build_index.py
│   ├── split_dataset.py
│   ├── dataset.py
│   ├── test_dataset.py
│   ├── visualize_dataset.py
│   ├── model.py
│   └── train.py
├── notebooks/
├── results/
│   ├── dataset_samples.png
│   └── checkpoints/
├── experiments/
├── tests/
├── .gitignore
└── README.md
```

The structure separates dataset preparation, model development, experiments, and evaluation. Raw datasets, virtual environments, and local model checkpoints should remain excluded from version control where appropriate.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/zMHA/XMedAI.git
cd XMedAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install torch torchvision pandas matplotlib scikit-learn opencv-python jupyter
```

### 4. Prepare the dataset

Place the dataset in the location expected by the dataset indexing scripts. Update the data path in the configuration or script if necessary.

Build the dataset index:

```bash
python src/build_index.py
```

Generate patient-level splits:

```bash
python src/split_dataset.py
```

Run the dataset validation script:

```bash
python src/test_dataset.py
```

### 5. Train the baseline model

```bash
python src/train.py
```

**Note:** Commands assume the scripts use these entry points and the dataset paths have been configured correctly. Check each script's arguments and configuration before running it.

## Reproducibility and Research Integrity

XMedAI is being developed as a research-oriented project. The following principles guide its implementation:

* Split data by patient rather than randomly assigning individual image slices.
* Keep training, validation, and test data separate.
* Record model configurations and evaluation metrics.
* Report validation and test results separately.
* Avoid making clinical claims based solely on experimental classification accuracy.
* Document limitations and investigate generalization before drawing conclusions.

## Current Status

* [x] Dataset indexing and metadata preparation
* [x] Patient-level training, validation, and test splits
* [x] Dataset loading and validation pipeline
* [x] ResNet18 baseline implementation
* [x] Initial classifier-head training
* [x] Initial validation result recorded
* [ ] Final independent test-set evaluation
* [ ] Detailed per-class metrics and confusion matrix
* [ ] Grad-CAM explainability experiments
* [ ] Robustness and generalization experiments
* [ ] Comparison with alternative model architectures

## Future Direction

The next stage is to evaluate the baseline on the held-out test set, analyze class-wise errors, and develop explainability experiments. Subsequent work will focus on whether the model relies on meaningful tumor-related image features and whether its performance remains consistent across unseen patients.

## Repository

GitHub: https://github.com/zMHA/XMedAI

## Disclaimer

XMedAI is an experimental research project and is not intended for clinical diagnosis or as a replacement for professional medical assessment.

## License

A license has not been specified yet. Add an appropriate open-source license before distributing the project for reuse.
