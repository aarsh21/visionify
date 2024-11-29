# Documentation: Training and Evaluating a YOLO Model on Custom Dataset in Google Colab
## [Google Colab](https://colab.research.google.com/drive/1tXxkF7vIynAsy4oiUagOCANZzjOB0_tw?usp=sharing)
This guide provides step-by-step instructions for training and evaluating a YOLO object detection model using a custom dataset in Google Colab. 

## Prerequisites

Before starting:
- Ensure you have a Google Drive account to store your dataset and outputs.
- Use the [Dataset.zip](https://drive.google.com/file/d/1tXxkF7vIynAsy4oiUagOCANZzjOB0_tw/view?usp=sharing) file provided for training. This dataset is pre-configured with relative paths compatible with Google Colab.

---

## Steps to Train and Evaluate the YOLO Model

### 1. **Mount Google Drive**
Mount your Google Drive to access the dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

This mounts your Google Drive at `/content/drive`, enabling access to stored files.

---

### 2. **Install YOLOv8 Dependencies**
Install the `ultralytics` library, which provides the YOLOv8 model.

```bash
!pip install ultralytics
```

---

### 3. **Extract the Dataset**
Unzip the provided dataset into a working directory.

```python
import zipfile
import os

# Path to the ZIP file
zip_file_path = "/content/drive/MyDrive/Dataset.zip"

# Path to extract the contents
extract_to_path = "/content/extracted_dataset"

# Ensure the target directory exists
os.makedirs(extract_to_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Files extracted to: {extract_to_path}")
```

This extracts the dataset to `/content/extracted_dataset`, ready for training.

---

### 4. **Load and Configure the YOLO Model**
Load a YOLOv8 model. Replace `yolo11l.pt` with the path to a pre-trained weight file if needed.

```python
from ultralytics import YOLO

# Load a YOLO model
model = YOLO('yolo11l.pt')  # Replace with your model's weight file path
```

---

### 5. **Train the Model**
Start training on the dataset. Ensure the `data.yaml` file in the dataset directory specifies correct paths.

```python
# Train the YOLO model
results = model.train(data='/content/extracted_dataset/Dataset/Dataset/data.yaml', epochs=20)
```

#### Key Training Parameters:
- `data`: Path to the `data.yaml` file specifying dataset classes and paths.
- `epochs`: Number of training epochs (default: 20).

---

### 6. **Evaluate the Model**
Evaluate the trained model on the validation dataset to compute performance metrics.

```python
# Evaluate the model
metrics = model.val()
```

---

### 7. **View Metrics**
#### Overall Metrics
Display mean Average Precision (mAP) and other metrics.

```python
print("Overall Metrics:")
print(f"mAP@50-95: {metrics.box.map:.4f}")  # Mean Average Precision (mAP) @ IoU 50-95
print(f"mAP@50: {metrics.box.map50:.4f}")  # mAP @ IoU 50
print(f"mAP@75: {metrics.box.map75:.4f}")  # mAP @ IoU 75
```

#### Per-Category Metrics
Display precision, recall, and mAP for each class.

```python
print("Per-Category Metrics:")
categories = metrics.names  # Access class names

for idx, category in categories.items():  # Iterate over class indices and names
    class_metrics = metrics.box.class_result(idx)  # Per-class metrics
    print(f"Category: {category}")
    print(f"  mAP@50-95: {class_metrics[1]:.4f}")  # Per-category mAP@50-95
    print(f"  Precision: {class_metrics[0]:.4f}")  # Per-category Precision
    print(f"  Recall: {class_metrics[2]:.4f}")  # Per-category Recall
    print()
```

---

## Output Summary
- **Training Results**: Saved in the `runs` directory under `/content`. It contains:
  - Training logs.
  - Model weights.
  - Visualizations of results.

- **Evaluation Metrics**: Includes overall mAP scores and per-class metrics for precision, recall, and mAP.

---

## Notes
- **Dataset Configuration**:
  Ensure that `data.yaml` has paths relative to the Colab working directory, as in the provided dataset.


