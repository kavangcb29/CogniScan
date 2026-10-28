#!/usr/bin/env python3
"""
Targeted Fine-Tuning Script for Alzheimer's Classifier
Freezes the pretrained EfficientNetB0 backbone and fine-tunes only the top dense classification head
on the pristine 4-class Alzheimer MRI dataset.
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
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from models.alzheimer_model import build_alzheimer_staging_model, ALZHEIMER_STAGE_CLASSES

CLASS_MAPPING = {
    "No Impairment": 0,
    "Very Mild Impairment": 1,
    "Mild Impairment": 2,
    "Moderate Impairment": 3,
}

def load_split(folder_path: Path):
    file_paths = []
    labels = []
    for class_folder, label_idx in CLASS_MAPPING.items():
        dir_path = folder_path / class_folder
        if not dir_path.exists():
            continue
        for f in dir_path.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                file_paths.append(str(f))
                labels.append(label_idx)
    return file_paths, labels

def create_tf_dataset(file_paths, labels, batch_size=32, is_training=True):
    def _parse(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Alzheimer Model Head")
    parser.add_argument("--train-dir", default=r"C:\Users\kavan\Documents\!College\Final project\FINAL\ALZHEIMERS MRI\Combined Dataset\train")
    parser.add_argument("--test-dir", default=r"C:\Users\kavan\Documents\!College\Final project\FINAL\ALZHEIMERS MRI\Combined Dataset\test")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-model", default="alz_model_finetuned.h5")
    args = parser.parse_args()

    train_path = Path(args.train_dir)
    test_path = Path(args.test_dir)

    print("=" * 70)
    print("🧠 ALZHEIMER MODEL HEAD FINE-TUNER")
    print("=" * 70)
    print(f"Train Dataset : {train_path}")
    print(f"Test Dataset  : {test_path}")

    train_files, train_labels = load_split(train_path)
    test_files, test_labels = load_split(test_path)

    print(f"✅ Discovered {len(train_files)} training images and {len(test_files)} testing images.")

    # Calculate class weights
    class_weights_arr = compute_class_weight("balanced", classes=np.unique(train_labels), y=np.array(train_labels))
    class_weights_dict = dict(enumerate(class_weights_arr))
    print(f"⚖️ Balanced class weights: {class_weights_dict}")

    train_ds = create_tf_dataset(train_files, train_labels, batch_size=args.batch_size, is_training=True)
    test_ds = create_tf_dataset(test_files, test_labels, batch_size=args.batch_size, is_training=False)

    print("\n🏗️ Building model and transferring EfficientNetB0 feature weights...")
    model = build_alzheimer_staging_model(variant="efficientnetb0")
    if os.path.exists("alz_model_improved_efficient.h5"):
        model.load_weights("alz_model_improved_efficient.h5")
        print("✅ Pretrained weights loaded from alz_model_improved_efficient.h5")

    # Freeze base model so only dense classification layers train
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n🚀 Fine-tuning classification head for {args.epochs} epochs...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output_model, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
    ]

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    print(f"\n📊 Final Evaluation on untouched test set:")
    y_preds_prob = model.predict(test_ds)
    y_preds = np.argmax(y_preds_prob, axis=1)

    print("\n📋 Classification Report:")
    report_dict = classification_report(test_labels, y_preds, target_names=ALZHEIMER_STAGE_CLASSES, output_dict=True)
    print(classification_report(test_labels, y_preds, target_names=ALZHEIMER_STAGE_CLASSES))

    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(test_labels, y_preds)
    print(cm)
    print(f"\n💾 Best model saved to: {args.output_model}")

    # Save metrics JSON & Confusion Matrix Plot
    out_dir = Path("test_results_finetuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump({
            "model": args.output_model,
            "accuracy": float(np.mean(np.array(test_labels) == y_preds)),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        acc = float(np.mean(np.array(test_labels) == y_preds))
        plt.title(f"Fine-Tuned Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
        plt.colorbar()
        tick_marks = np.arange(len(ALZHEIMER_STAGE_CLASSES))
        plt.xticks(tick_marks, ALZHEIMER_STAGE_CLASSES, rotation=45, ha="right")
        plt.yticks(tick_marks, ALZHEIMER_STAGE_CLASSES)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True Stage")
        plt.xlabel("Predicted Stage")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
        plt.close()
        print(f"📈 Confusion Matrix plot saved: {out_dir / 'confusion_matrix.png'}")
    except Exception as ex:
        pass

if __name__ == "__main__":
    main()
