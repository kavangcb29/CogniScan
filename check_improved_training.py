"""
Check Improved Brain Tumor Model Training Results
"""

import os
import sys
import json
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def check_training_results():
    """Check the results of the improved training."""

    models_dir = Path("trained_models")
    improved_model = models_dir / "brain_tumor_cnn_multiclass_improved.h5"
    improved_metadata = models_dir / "tumor_model_multiclass_improved_metadata.json"

    print("=" * 70)
    print("IMPROVED BRAIN TUMOR MODEL TRAINING RESULTS")
    print("=" * 70)

    metadata = None
    if improved_model.exists():
        size_mb = improved_model.stat().st_size / (1024 * 1024)
        print(f"✓ Improved model saved: {improved_model}")
        print(f"  Size: {size_mb:.2f} MB")
        if improved_metadata.exists():
            try:
                with open(improved_metadata, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                print("\n📊 TRAINING RESULTS:")
                print(f"  Final Training Accuracy: {metadata['final_train_accuracy']:.4f}")
                print(f"  Final Validation Accuracy: {metadata['final_val_accuracy']:.4f}")
                print(f"  Final Test Accuracy: {metadata['final_test_accuracy']:.4f}")
                print(f"  Final Training Loss: {metadata['final_train_loss']:.4f}")
                print(f"  Final Validation Loss: {metadata['final_val_loss']:.4f}")
                print(f"  Final Test Loss: {metadata['final_test_loss']:.4f}")
                print(f"  Total Parameters: {metadata['total_parameters']:,}")
                print(f"  Training Samples: {metadata['training_samples']}")
                print(f"  Validation Samples: {metadata['validation_samples']}")
                print(f"  Test Samples: {metadata['test_samples']}")

                # Check if validation accuracy > 96%
                val_acc = metadata['final_val_accuracy']
                if val_acc > 0.96:
                    print("\n🎉 SUCCESS! Validation accuracy > 96%")
                    print(f"  Achieved: {val_acc:.1%}")
                else:
                    print("\n⚠️  Validation accuracy below 96%")
                    print(f"  Achieved: {val_acc:.1%}")

                print("\n🔧 IMPROVEMENTS APPLIED:")
                for improvement in metadata.get('improvements', []):
                    print(f"  • {improvement}")

            except Exception as e:
                print(f"❌ Error reading metadata: {e}")
        else:
            print("⏳ Metadata not yet saved (training may still be running)")
    else:
        print("⏳ Improved model not yet saved (training in progress)")

    # Compare with original model
    original_model = models_dir / "brain_tumor_cnn_multiclass.h5"
    original_metadata = models_dir / "tumor_model_multiclass_metadata.json"

    if original_model.exists() and original_metadata.exists():
        try:
            with open(original_metadata, 'r', encoding='utf-8') as f:
                orig_meta = json.load(f)

            print("\n📊 COMPARISON WITH ORIGINAL MODEL:")
            print(f"  Original Validation Accuracy: {orig_meta['final_val_accuracy']:.4f}")
            if metadata and 'final_val_accuracy' in metadata:
                print(f"  Improved Validation Accuracy: {metadata['final_val_accuracy']:.4f}")
                val_improvement = metadata['final_val_accuracy'] - orig_meta['final_val_accuracy']
                print(f"  Improvement: {val_improvement:.4f}")
                if val_improvement > 0:
                    print("  📈 Validation accuracy IMPROVED!")
                elif val_improvement < 0:
                    print("  📉 Validation accuracy decreased")
                else:
                    print("  ➡️  Validation accuracy same")

        except Exception as e:
            print(f"❌ Error comparing models: {e}")

if __name__ == "__main__":
    check_training_results()
