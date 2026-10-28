# CogniScan

CogniScan is a comprehensive deep learning medical imaging platform engineered primarily for Alzheimer's Disease cognitive staging and early biomarker detection, with auxiliary screening for brain tumor detection. It includes patient and doctor dashboards, MRI scan upload and automated multi-stage classification, clinical report generation, appointment scheduling, and communication tools.

## Features

- **Primary Focus**: Alzheimer's Disease multi-stage cognitive classification (Non-Demented, Very Mild, Mild, Moderate) via EfficientNetB0
- **Auxiliary Screener**: 4-class Brain Tumor classification via Deep CNN
- MRI preprocessing, enhancement, and segmentation bounding-box visualization
- Patient and Doctor clinical decision support portals
- Standardized medical PDF & printable neurodiagnostic report generation
- Appointment scheduling and consultation management
- Fully integrated Django and Flask architectures

## Tech Stack

- Python
- Flask
- Flask-SocketIO
- TensorFlow / Keras
- OpenCV
- SQLite
- Bootstrap

## Project Structure

```text
app.py                 Flask application entry point
routes/                API and feature routes
services/              Core business logic
models/                Model loading and inference helpers
training/              Training utilities and scripts
templates/             Patient/doctor HTML templates
static/                Static assets
tests/                 Test suite
```

## Getting Started

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the application

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Training

Tumor and Alzheimer training scripts live in `training/`.

Example:

```powershell
.\.venv-fresh\Scripts\python.exe training\train_tumor_model.py --backbone vgg16 --image-size 160 --batch-size 2 --epochs 30 --validation-split 0.25 --fine-tune-layers 4 --learning-rate 1e-5
```

## Notes

- Runtime files, uploads, datasets, logs, and local training outputs are excluded from Git.
- The repository currently includes baseline `.h5` model files used by the app at runtime.
