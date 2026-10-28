#!/usr/bin/env python3
"""
Unified Alzheimer Model Trainer & Fine-Tuner
Allows training or fine-tuning Alzheimer MRI models on any new dataset with custom parameters:
  - Custom data path (single dataset with split or separate train/test folders)
  - Backbone architecture (EfficientNetB0, MobileNetV2, ResNet50, or CNN)
  - Hyperparameters (Epochs, Batch Size, Learning Rate, Fine-tune Layers, Dropout, Dense Units)
  - Class weighting and data augmentation controls
  - Option to automatically activate the new model in model_manifest.json
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
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from models.alzheimer_model import build_alzheimer_staging_model, ALZHEIMER_STAGE_CLASSES

CLASS_ALIASES = {
    # NonDemented
    "nondemented": 0, "non demented": 0, "no impairment": 0, "no_impairment": 0, "healthy": 0, "non": 0, "0": 0,
    # Very Mild
    "very mild": 1, "very_mild": 1, "verymild": 1, "very mild impairment": 1, "very_mild_impairment": 1, "mci": 1, "1": 1,
    # Mild
    "mild": 2, "mild impairment": 2, "mild_impairment": 2, "mild dementia": 2, "2": 2,
    # Moderate
    "moderate": 3, "moderate impairment": 3, "moderate_impairment": 3, "severe": 3, "3": 3,
}

def resolve_folder_label(name: str):
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    if cleaned in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned]
    if "very mild" in cleaned:
        return 1
    if "no impairment" in cleaned or "non dement" in cleaned:
        return 0
    if "moderate" in cleaned or "severe" in cleaned:
        return 3
    if "mild" in cleaned:
        return 2
    for k in sorted(CLASS_ALIASES.keys(), key=len, reverse=True):
        if k in cleaned:
            return CLASS_ALIASES[k]
    return None

def scan_folder(folder_path: Path):
    file_paths = []
    labels = []
    for entry in folder_path.iterdir():
        if entry.is_dir():
            idx = resolve_folder_label(entry.name)
            if idx is None:
                continue
            for f in entry.rglob("*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    file_paths.append(str(f))
                    labels.append(idx)
    return np.array(file_paths), np.array(labels)

def create_dataset(file_paths, labels, batch_size=32, img_size=(224, 224), augment=True):
    def _parse(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def _augment(img, label):
        img = tf.image.random_flip_left_right(img)
        crop_size = tf.random.uniform([], int(img_size[0] * 0.9), img_size[0], dtype=tf.int32)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
        img = tf.image.resize(img, img_size)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser(description="Retrain/Fine-Tune Alzheimer Model on New Data")
    # Dataset args
    parser.add_argument("--data-dir", help="Path to dataset containing class subfolders (auto-splits train/val)")
    parser.add_argument("--train-dir", help="Path to training folder (if already split)")
    parser.add_argument("--val-dir", help="Path to validation/test folder (if already split)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio if using --data-dir (default: 0.2)")

    # Architecture & Transfer Learning
    parser.add_argument("--variant", choices=["efficientnetb0", "mobilenetv2", "resnet50", "vgg16", "cnn"], default="efficientnetb0", help="Model backbone architecture")
    parser.add_argument("--pretrained-weights", default="alz_model_improved_efficient.h5", help="Path to pretrained weights to start from, or 'imagenet', or 'none'")
    parser.add_argument("--fine-tune-layers", type=int, default=50, help="Number of backbone layers to unfreeze (0 = freeze entire backbone, 50 = top 50, -1 = full unfreeze)")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--dense-units", type=int, default=512, help="Dense units in classification head (default: 512)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate (default: 0.4)")
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced", help="Class weighting to handle imbalance")

    # Output & Deployment
    parser.add_argument("--output-model", default="alz_model_retrained.h5", help="File name to save the best model")
    parser.add_argument("--activate", action="store_true", help="Automatically activate this model in models/model_manifest.json upon completion")

    args = parser.parse_args()

    print("=" * 70)
    print("🧠 ALZHEIMER MODEL RETRAINING & PARAMETER CUSTOMIZATION")
    print("=" * 70)
    print(f"Architecture    : {args.variant}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch Size      : {args.batch_size}")
    print(f"Learning Rate   : {args.lr}")
    print(f"Dense Units     : {args.dense_units}")
    print(f"Dropout Rate    : {args.dropout}")
    print(f"Fine-tune Layers: {args.fine_tune_layers}")
    print(f"Output Model    : {args.output_model}")
    print("=" * 70)

    # Load data
    if args.train_dir and args.val_dir:
        print(f"Loading train directory: {args.train_dir}")
        train_paths, train_labels = scan_folder(Path(args.train_dir))
        print(f"Loading val directory: {args.val_dir}")
        val_paths, val_labels = scan_folder(Path(args.val_dir))
    elif args.data_dir:
        print(f"Loading dataset from: {args.data_dir}")
        all_paths, all_labels = scan_folder(Path(args.data_dir))
        if len(all_paths) == 0:
            print(f"❌ Error: No valid image files found in {args.data_dir}")
            sys.exit(1)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=args.val_split, stratify=all_labels, random_state=42
        )
    else:
        print("❌ Error: Must specify either --data-dir OR both --train-dir and --val-dir")
        sys.exit(1)

    print(f"\n📊 Data Summary:")
    print(f"   • Training Scans   : {len(train_paths)}")
    print(f"   • Validation Scans : {len(val_paths)}")
    for idx, cname in enumerate(ALZHEIMER_STAGE_CLASSES):
        tr_cnt = np.sum(train_labels == idx)
        vl_cnt = np.sum(val_labels == idx)
        print(f"   • {cname:15s}: Train={tr_cnt:5d} | Val={vl_cnt:5d}")

    # Build model
    print(f"\n🏗️ Building {args.variant} model...")
    img_size = (224, 224)
    model = build_alzheimer_staging_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=4,
        variant=args.variant,
        weights="imagenet" if args.pretrained_weights.lower() == "imagenet" else None,
        dense_units=args.dense_units,
        dropout_rate=args.dropout,
        fine_tune_layers=args.fine_tune_layers
    )

    # Load pretrained weights if requested
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        print(f"📥 Loading pretrained weights from: {args.pretrained_weights}")
        try:
            model.load_weights(args.pretrained_weights, by_name=True)
            print("✅ Pretrained weights transferred successfully!")
        except Exception as e:
            print(f"⚠️ Pretrained weight transfer notice: {e}")

    # Fine-tune layer freezing control
    if args.fine_tune_layers == 0:
        print("🔒 Freezing all backbone layers (training classification head only)")
        for layer in model.layers:
            if args.variant.lower() in layer.name.lower():
                layer.trainable = False
    elif args.fine_tune_layers > 0:
        print(f"🔓 Fine-tuning top {args.fine_tune_layers} backbone layers")

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Class weights
    cw_dict = None
    if args.class_weights == "balanced":
        train_counts = np.bincount(train_labels)
        # If training data is uniformly oversampled (equal count per class), align with target validation distribution
        if len(set(train_counts)) == 1 and len(val_labels) > 0:
            print("⚖️ Note: Training set is uniformly oversampled; balancing against real-world test distribution...")
            cw_arr = compute_class_weight("balanced", classes=np.unique(val_labels), y=val_labels)
        else:
            cw_arr = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        cw_dict = dict(enumerate(cw_arr))
        print(f"⚖️ Applied balanced class weights: {cw_dict}")

    # Datasets
    train_ds = create_dataset(train_paths, train_labels, batch_size=args.batch_size, img_size=img_size, augment=True)
    val_ds = create_dataset(val_paths, val_labels, batch_size=args.batch_size, img_size=img_size, augment=False)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output_model, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=cw_dict,
        callbacks=callbacks
    )

    # Evaluate best model
    print("\n🔍 Evaluating best model on validation data...")
    try:
        model.load_weights(args.output_model)
    except Exception:
        pass

    y_pred_probs = model.predict(val_ds)
    y_preds = np.argmax(y_pred_probs, axis=1)

    print("\n📋 Final Classification Report:")
    print(classification_report(val_labels, y_preds, target_names=ALZHEIMER_STAGE_CLASSES, zero_division=0))

    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(val_labels, y_preds)
    print(cm)

    # Save metrics JSON
    metrics_path = Path(args.output_model).stem + "_metrics.json"
    rep_dict = classification_report(val_labels, y_preds, target_names=ALZHEIMER_STAGE_CLASSES, output_dict=True, zero_division=0)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.output_model,
            "architecture": args.variant,
            "epochs_completed": len(history.history["accuracy"]),
            "final_val_accuracy": float(np.mean(val_labels == y_preds)),
            "confusion_matrix": cm.tolist(),
            "classification_report": rep_dict,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"💾 Metrics summary saved to: {metrics_path}")

    # Activate if requested
    if args.activate:
        manifest_path = Path("models/model_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["alzheimer_classifier"]["path"] = args.output_model
            manifest["alzheimer_classifier"]["metadata_path"] = metrics_path
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"🎉 Model activated! Updated models/model_manifest.json to point to {args.output_model}")

if __name__ == "__main__":
    main()
