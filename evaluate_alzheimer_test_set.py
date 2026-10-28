#!/usr/bin/env python3
"""
Alzheimer Model Test Set Evaluator
Comprehensive evaluation script for testing the 4-class Alzheimer MRI model.
Generates:
  - Overall Accuracy, Loss, Precision, Recall, and F1-Score
  - Confusion Matrix (printed and saved as image)
  - Full Classification Report per stage
  - Detailed predictions CSV log for every test image
"""

import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

# Canonical target classes used by the model
TARGET_CLASSES = ["NonDemented", "Very Mild", "Mild", "Moderate"]

# Robust class folder aliases mapping to canonical index (0 to 3)
CLASS_ALIASES = {
    # Class 0: NonDemented
    "nondemented": 0,
    "non demented": 0,
    "non_demented": 0,
    "no impairment": 0,
    "no_impairment": 0,
    "notdemented": 0,
    "healthy": 0,
    "normal": 0,
    "non": 0,
    "0": 0,

    # Class 1: Very Mild
    "very mild": 1,
    "very_mild": 1,
    "verymild": 1,
    "very mild dementia": 1,
    "very_mild_dementia": 1,
    "verymilddementia": 1,
    "very mild impairment": 1,
    "very_mild_impairment": 1,
    "mci": 1,
    "1": 1,

    # Class 2: Mild
    "mild": 2,
    "mild dementia": 2,
    "mild_dementia": 2,
    "milddementia": 2,
    "mild impairment": 2,
    "mild_impairment": 2,
    "2": 2,

    # Class 3: Moderate
    "moderate": 3,
    "moderate dementia": 3,
    "moderate_dementia": 3,
    "moderatedementia": 3,
    "moderate impairment": 3,
    "moderate_impairment": 3,
    "severe": 3,
    "severe impairment": 3,
    "severe dementia": 3,
    "3": 3,
}

def normalize_folder_name(name: str):
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    cleaned_compact = cleaned.replace(" ", "")
    
    # 1. Exact matches first
    if cleaned in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned]
    if cleaned_compact in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned_compact]

    # 2. Check compound phrases first (e.g. 'very mild', 'no impairment')
    if "very mild" in cleaned:
        return 1
    if "no impairment" in cleaned or "non dement" in cleaned or "not dement" in cleaned:
        return 0
    if "moderate" in cleaned or "severe" in cleaned:
        return 3
    if "mild" in cleaned:
        return 2

    # 3. Sort keys by length descending to match most specific key first
    for key in sorted(CLASS_ALIASES.keys(), key=len, reverse=True):
        if key in cleaned:
            return CLASS_ALIASES[key]

    return None

def scan_dataset(data_dir: Path):
    """Scans dataset directory for class subfolders or images."""
    samples = []
    unmapped_folders = []

    for entry in data_dir.iterdir():
        if entry.is_dir():
            class_idx = normalize_folder_name(entry.name)
            if class_idx is None:
                unmapped_folders.append(entry.name)
                continue
            
            for file_path in entry.rglob("*"):
                if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    samples.append((str(file_path), class_idx, entry.name))

    return samples, unmapped_folders

def preprocess_image(image_path: str, target_size=(224, 224)):
    """Loads and preprocesses image to model format (224x224x3, [0, 1])."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def main():
    parser = argparse.ArgumentParser(description="Evaluate Alzheimer Model on Test Dataset")
    parser.add_argument("--data-dir", required=True, help="Path to testing dataset directory with class subfolders")
    parser.add_argument("--model-path", default="alz_model_improved_efficient.h5", help="Path to trained .h5 model")
    parser.add_argument("--output-dir", default="test_results", help="Directory to save evaluation reports and metrics")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"❌ Error: Dataset directory '{data_dir}' not found!")
        sys.exit(1)

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"❌ Error: Model file '{model_path}' not found!")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🧠 ALZHEIMER MODEL TEST DATASET EVALUATOR")
    print("=" * 70)
    print(f"📁 Dataset Directory : {data_dir}")
    print(f"🤖 Model File        : {model_path}")
    print(f"📊 Output Directory  : {output_dir}")

    # Discover samples
    samples, unmapped = scan_dataset(data_dir)
    if not samples:
        print(f"❌ No valid images found under recognized class folders in '{data_dir}'!")
        if unmapped:
            print(f"⚠️  Found unmapped folders: {unmapped}")
            print(f"💡 Expected folder names: NonDemented, Very Mild, Mild, Moderate")
        sys.exit(1)

    counts = defaultdict(int)
    for _, idx, _ in samples:
        counts[idx] += 1

    print(f"\n✅ Total test images discovered: {len(samples)}")
    print("📋 Class Distribution:")
    for idx, cname in enumerate(TARGET_CLASSES):
        print(f"   • {cname:15s}: {counts[idx]} images")

    # Load TensorFlow & model
    print("\n⏳ Loading TensorFlow and model...")
    try:
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        try:
            import keras
            if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
                keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
    except ImportError as e:
        print(f"❌ Required package missing: {e}")
        print("Please ensure tensorflow and scikit-learn are installed.")
        sys.exit(1)

    try:
        from models.alzheimer_model import build_alzheimer_staging_model
        print("🏗️  Building EfficientNetB0 architecture and loading trained weights...")
        model = build_alzheimer_staging_model(variant="efficientnetb0")
        model.load_weights(str(model_path))
        print("✅ Model weights loaded successfully!")
    except Exception as e1:
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            print("✅ Model loaded successfully via load_model!")
        except Exception as e2:
            print(f"❌ Error loading model: {e1} / {e2}")
            sys.exit(1)

    # Run inference in batches
    print("\n🔍 Running inference on test dataset...")
    y_true = []
    y_pred = []
    y_probs = []
    detailed_records = []

    batch_size = args.batch_size
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_images = []
        batch_labels = []

        for fpath, label_idx, folder_name in batch_samples:
            try:
                arr = preprocess_image(fpath)
                batch_images.append(arr)
                batch_labels.append((fpath, label_idx, folder_name))
            except Exception as ex:
                print(f"⚠️ Failed to process image {fpath}: {ex}")

        if not batch_images:
            continue

        batch_arr = np.array(batch_images)
        preds = model.predict(batch_arr, verbose=0)

        for (fpath, true_idx, folder_name), prob in zip(batch_labels, preds):
            pred_idx = int(np.argmax(prob))
            conf = float(prob[pred_idx])

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(prob.tolist())

            detailed_records.append({
                "file_path": fpath,
                "file_name": os.path.basename(fpath),
                "original_folder": folder_name,
                "true_label": TARGET_CLASSES[true_idx],
                "predicted_label": TARGET_CLASSES[pred_idx],
                "confidence": round(conf * 100, 2),
                "is_correct": (true_idx == pred_idx),
                "prob_NonDemented": round(float(prob[0]) * 100, 2),
                "prob_VeryMild": round(float(prob[1]) * 100, 2),
                "prob_Mild": round(float(prob[2]) * 100, 2),
                "prob_Moderate": round(float(prob[3]) * 100, 2),
            })

        progress = min(i + batch_size, len(samples))
        print(f"   Processed {progress}/{len(samples)} images ({progress * 100 // len(samples)}%)", end="\r")

    print(f"\n✅ Completed inference on all {len(y_true)} images!")

    # Calculate overall metrics
    correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total_count = len(y_true)
    accuracy = (correct_count / total_count) if total_count > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    report_dict = classification_report(y_true, y_pred, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=TARGET_CLASSES, digits=4, zero_division=0)

    print("\n" + "=" * 70)
    print(f"🎯 EVALUATION SUMMARY: Overall Accuracy = {accuracy * 100:.2f}% ({correct_count}/{total_count})")
    print("=" * 70)
    print("\n📋 CLASSIFICATION REPORT:")
    print(report_text)

    print("\n🔢 CONFUSION MATRIX:")
    header = f"{'True \\ Pred':15s}" + "".join(f"{c[:10]:>12s}" for c in TARGET_CLASSES)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{TARGET_CLASSES[i]:15s}" + "".join(f"{val:12d}" for val in row)
        print(row_str)

    # Save CSV of detailed records
    csv_path = output_dir / "predictions_log.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file_name", "original_folder", "true_label", "predicted_label",
            "is_correct", "confidence", "prob_NonDemented", "prob_VeryMild",
            "prob_Mild", "prob_Moderate", "file_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_records)

    # Save metrics JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    results_payload = {
        "dataset_directory": str(data_dir),
        "total_test_samples": total_count,
        "correct_predictions": correct_count,
        "accuracy": round(accuracy, 4),
        "accuracy_percentage": f"{accuracy * 100:.2f}%",
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classes": TARGET_CLASSES
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Try plotting confusion matrix if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
        plt.colorbar()
        tick_marks = np.arange(len(TARGET_CLASSES))
        plt.xticks(tick_marks, TARGET_CLASSES, rotation=45, ha="right")
        plt.yticks(tick_marks, TARGET_CLASSES)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True Stage")
        plt.xlabel("Predicted Stage")
        plt.tight_layout()
        plot_path = output_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"📈 Confusion Matrix plot saved: {plot_path}")
    except Exception as ex:
        pass

    print(f"\n💾 Results exported successfully:")
    print(f"   • Metrics JSON : {metrics_path}")
    print(f"   • Detailed CSV : {csv_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
