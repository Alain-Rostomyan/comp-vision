# Computer Vision: Emoji Classification

This repository contains code for classifying emoji images into vendors (Apple, Google, Facebook, etc.).

## Structure

- **`common/`**: Shared utilities, configuration, and dataset classes.
- **`train_baseline.py`**: Trains a simple CNN from scratch (Day 1).
- **`train_transfer_learning.py`**: Trains a ResNet18 model using Transfer Learning (Day 2).
- **`inference_tta.py`**: Generates high-accuracy predictions using Test-Time Augmentation (Day 5).
- **`src/`**: Legacy reference code (Deprecated).

## Setup

1. **Create a Virtual Environment**
   ```bash
   python -m venv venv

   #Using uv 
   uv venv
   ```

2. **Activate the Environment**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

   #Using uv 
   uv pip install -r requirements.txt
   ```

## How to Run

### 1. Train Baseline Model (Fast)
```bash
python train_baseline.py

#Using uv 
uv run train_baseline.py
```
*Outputs: `outputs/submission_baseline.csv`*

### 2. Train High-Performance Model (Transfer Learning)
```bash
python train_transfer_learning.py

#Using uv 
uv run train_transfer_learning.py
```
*Outputs: `outputs/submission_transfer.csv`, `outputs/transfer_best_model.pth`*

### 3. Generate Final Predictions (with TTA)
This uses your trained model (from step 2) to generate the most accurate predictions possible.
```bash
python inference_tta.py

#Using uv 
uv run inference_tta.py
```
*Outputs: `outputs/submission_tta.csv`*