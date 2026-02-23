#!/usr/bin/env python3
"""
YOLOv8-pose Training Script for microPAD Quadrilateral Corner Detection

QUICK START:
    python train_yolo.py

TRAINING PIPELINE:
- Stage 1: Pretraining on synthetic data
- Stage 2: Fine-tuning on mixed synthetic + manual labels (future)

Also provides validation and export capabilities for deployment.

PERFORMANCE TARGETS:
- OKS mAP@50 > 0.85
- Detection rate > 85%
- Inference < 100ms per image
"""

# GPU Configuration (configured by /configure_training_gpus)
# Selected: 2x RTX A6000 with NVLink (96GB combined VRAM)
# PyTorch device mapping (differs from nvidia-smi due to PCI bus ordering):
#   PyTorch cuda:0 -> NVIDIA RTX A6000 (48GB) [nvidia-smi GPU 0]
#   PyTorch cuda:1 -> NVIDIA RTX A6000 (48GB) [nvidia-smi GPU 2] - NVLinked
#   PyTorch cuda:2 -> NVIDIA RTX 3090 (24GB)  [nvidia-smi GPU 1]
# MUST be set before importing torch/ultralytics
import os
# Only set if not already configured (allows external override)
# Note: Uses PyTorch indices (0,1 = both A6000s), not nvidia-smi indices
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import yaml

try:
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings
    from ultralytics import __version__ as ultralytics_version
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ============================================================================
# TRAINING CONFIGURATION CONSTANTS
# ============================================================================
#
# ZERO-CONFIGURATION TRAINING:
# These defaults are optimized for dual RTX A6000 workstation with NVLink (96 GB VRAM total)
# with large input images (~13MP: 4032x3024 → resized to 1280x1280).
# Simply run: python train_yolo.py
# No command-line arguments required for optimal performance!
#
# Batch size is set to 16 (8 per GPU) to handle large input images safely.
# To customize: Override any parameter via CLI flags (see --help)
# ============================================================================

# ============================================================================
# 3-TIER PRESET SYSTEM (YOLOv8)
# ============================================================================
# Presets simplify training configuration. Each preset optimizes model size,
# resolution, and batch size for specific use cases.
#
# | Preset    | Model          | Resolution | Batch | Use Case                |
# |-----------|----------------|------------|-------|-------------------------|
# | medium    | yolov8m-pose   | 640px      | 32    | DEFAULT - High accuracy |
# | small     | yolov8s-pose   | 640px      | 48    | Balanced speed/accuracy |
# | nano      | yolov8n-pose   | 640px      | 64    | Fast training/inference |
# ============================================================================

# Medium preset (DEFAULT) - high accuracy for production
PRESET_MEDIUM_MODEL = 'yolov8m-pose.pt'
PRESET_MEDIUM_IMGSZ = 640
PRESET_MEDIUM_BATCH = 32       # 16 per GPU

# Small preset - balanced speed and accuracy
PRESET_SMALL_MODEL = 'yolov8s-pose.pt'
PRESET_SMALL_IMGSZ = 640
PRESET_SMALL_BATCH = 48        # 24 per GPU

# Nano preset - fast training/inference
PRESET_NANO_MODEL = 'yolov8n-pose.pt'
PRESET_NANO_IMGSZ = 640
PRESET_NANO_BATCH = 64         # 32 per GPU

# Default experiment names for each preset
DEFAULT_NAME_MEDIUM = 'yolov8m_pose_640'
DEFAULT_NAME_SMALL = 'yolov8s_pose_640'
DEFAULT_NAME_NANO = 'yolov8n_pose_640'

# Legacy defaults (for backward compatibility in class methods)
DEFAULT_MODEL = PRESET_MEDIUM_MODEL
DEFAULT_IMAGE_SIZE = PRESET_MEDIUM_IMGSZ

# Training hyperparameters (optimized for dual A6000 with NVLink)
# NOTE: Batch size conservative due to large input images (~13MP: 4032x3024)
DEFAULT_BATCH_SIZE = PRESET_MEDIUM_BATCH  # 16 per GPU - safe for large images at 640 resolution
DEFAULT_EPOCHS_STAGE1 = 200          # Extended training for better convergence
DEFAULT_EPOCHS_STAGE2 = 150          # Extended fine-tuning with early stopping
DEFAULT_PATIENCE_STAGE1 = 20         # Early stopping patience
DEFAULT_PATIENCE_STAGE2 = 15
DEFAULT_LEARNING_RATE_STAGE1 = 0.001 # Learning rate for AdamW optimizer (stage 1)
DEFAULT_LEARNING_RATE_STAGE2 = 0.0005 # Lower LR for fine-tuning
DEFAULT_OPTIMIZER = 'AdamW'          # AdamW optimizer for pose estimation

# Hardware configuration (optimized for dual RTX A6000 with NVLink)
# GPU device selection: CUDA_VISIBLE_DEVICES is set at script top to "0,1"
# Both GPUs are RTX A6000 (48GB each, 96GB total with NVLink interconnect)
DEFAULT_GPU_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0,1')
DEFAULT_NUM_WORKERS = 16             # Optimal for data loading
DEFAULT_CACHE_ENABLED = 'disk'       # Disk cache for deterministic training

# Checkpoint configuration
CHECKPOINT_SAVE_PERIOD = 10          # Save checkpoint every N epochs

# Augmentation configuration (server/desktop)
AUG_HSV_HUE = 0.015
AUG_HSV_SATURATION = 0.7
AUG_HSV_VALUE = 0.4
AUG_TRANSLATE = 0.1
AUG_SCALE = 0.5
AUG_FLIP_LR = 0.5
AUG_MOSAIC = 0.8
AUG_ROTATION = 0.0  # Disabled (already in synthetic data)
AUG_ERASING = 0.4  # Random erasing for occlusion robustness
AUG_MIXUP = 0.0  # Disabled by default, can enable via --mixup

# Mobile augmentation (reduced for lower resolution - features are smaller)
AUG_SCALE_MOBILE = 0.3
AUG_TRANSLATE_MOBILE = 0.05
AUG_MOSAIC_MOBILE = 0.8

# Dataset configuration
DEFAULT_STAGE1_DATA = 'micropad_synth.yaml'
DEFAULT_STAGE2_DATA = 'micropad_mixed.yaml'

# ============================================================================

# Verify Ultralytics version supports YOLOv8 pose training
def check_ultralytics_version() -> None:
    """Check that Ultralytics version supports YOLOv8-pose."""
    required_version = (8, 0, 0)  # YOLOv8 requires ultralytics >= 8.0.0
    try:
        version_parts = tuple(int(x) for x in ultralytics_version.split('.')[:3])
        if version_parts < required_version:
            print(f"ERROR: Ultralytics version {ultralytics_version} does not support YOLOv8-pose.")
            print(f"       Required version: {'.'.join(map(str, required_version))} or higher")
            print(f"       Update with: pip install --upgrade ultralytics")
            sys.exit(1)
    except Exception:
        print(f"WARNING: Could not parse Ultralytics version: {ultralytics_version}")

check_ultralytics_version()


class YOLOTrainer:
    """YOLOv8 training pipeline for microPAD auto-detection."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize trainer.

        Args:
            project_root: Path to project root. Auto-detected if None.
        """
        if project_root is None:
            # Auto-detect project root (search up to 5 levels)
            current = Path(__file__).resolve().parent
            for _ in range(5):
                if (current / 'CLAUDE.md').exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path(__file__).resolve().parent.parent

        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / 'python_scripts' / 'configs'
        self.results_dir = self.project_root / 'training_runs'

        # Verify project structure
        if not self.configs_dir.exists():
            raise FileNotFoundError(
                f"Configs directory not found: {self.configs_dir}\n"
                f"Expected structure: {self.project_root}/python_scripts/configs/"
            )

    def train_stage1(
        self,
        model: str = DEFAULT_MODEL,
        data: str = DEFAULT_STAGE1_DATA,
        epochs: int = DEFAULT_EPOCHS_STAGE1,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        batch: int = DEFAULT_BATCH_SIZE,
        device: str = DEFAULT_GPU_DEVICES,
        patience: int = DEFAULT_PATIENCE_STAGE1,
        workers: int = DEFAULT_NUM_WORKERS,
        cache: Union[bool, str] = DEFAULT_CACHE_ENABLED,
        optimizer: str = DEFAULT_OPTIMIZER,
        lr0: float = DEFAULT_LEARNING_RATE_STAGE1,
        name: str = 'yolov8m_pose_640',
        **kwargs
    ) -> Dict[str, Any]:
        """Train Stage 1: Synthetic data pretraining.

        Args:
            model: YOLO pretrained model for keypoint detection (default: yolov8m-pose.pt)
            data: Dataset config file in configs/ directory
            epochs: Maximum training epochs
            imgsz: Input image size
            batch: Batch size (total across all GPUs)
            device: GPU device(s) (e.g., '0' for single GPU, '0,1' for multi-GPU)
            patience: Early stopping patience
            workers: Number of dataloader workers
            cache: Cache images in RAM/disk for faster training (True, False, or 'disk')
            optimizer: Optimizer type (default: AdamW)
            lr0: Initial learning rate (default: 0.001 for AdamW)
            name: Experiment name
            **kwargs: Additional training arguments passed to YOLO

        Returns:
            Training results dictionary
        """
        print("\n" + "="*80)
        print("STAGE 1: SYNTHETIC DATA PRETRAINING")
        print("="*80)

        # Verify label directories exist before training
        print("\nVerifying label directories...")
        try:
            from prepare_yolo_dataset import verify_label_directories, discover_phone_directories
            phone_dirs = discover_phone_directories()
            if not verify_label_directories(phone_dirs):
                print("  WARNING: Some label directories missing. Run prepare_yolo_dataset.py first.")
        except ImportError:
            print("  WARNING: prepare_yolo_dataset not found, skipping directory verification")
        except Exception as e:
            print(f"  WARNING: Failed to verify label directories: {e}")

        # Resolve data config path
        data_path = self.configs_dir / data
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset config not found: {data_path}\n"
                f"Run prepare_yolo_dataset.py first to generate config."
            )

        num_devices = len(device.split(',')) if ',' in device else 1
        batch_per_device = batch // num_devices if num_devices > 1 else batch

        print(f"\nConfiguration:")
        print(f"  Model: {model}")
        print(f"  Data: {data_path}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}" + (f" ({batch_per_device} per GPU)" if num_devices > 1 else ""))
        print(f"  Optimizer: {optimizer}")
        print(f"  Learning rate: {lr0}")
        print(f"  Workers: {workers}")
        print(f"  Cache: {cache}")
        print(f"  Device(s): {device}")
        print(f"  Patience: {patience}")
        print(f"  Project: {self.results_dir}")
        print(f"  Name: {name}")

        # Load model
        print(f"\nLoading model: {model}")
        yolo_model = YOLO(model)

        # Training arguments
        train_args = {
            'data': str(data_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'workers': workers,
            'cache': cache,
            'project': str(self.results_dir),
            'name': name,
            'patience': patience,
            'save': True,
            'save_period': CHECKPOINT_SAVE_PERIOD,
            'verbose': True,
            'plots': True,
            # Optimizer configuration
            'optimizer': optimizer,
            'lr0': lr0,
            'cos_lr': True,  # Cosine learning rate scheduler
            'amp': True,  # Automatic Mixed Precision for faster training
            # Pose-specific loss weights (optimized for keypoint detection)
            'pose': 12.0,  # Pose loss weight
            'kobj': 2.0,   # Keypoint objectness loss weight
            # Augmentation configuration
            'hsv_h': AUG_HSV_HUE,
            'hsv_s': AUG_HSV_SATURATION,
            'hsv_v': AUG_HSV_VALUE,
            'translate': AUG_TRANSLATE,
            'scale': AUG_SCALE,
            'fliplr': AUG_FLIP_LR,
            'mosaic': AUG_MOSAIC,
            'degrees': AUG_ROTATION,
            'erasing': AUG_ERASING,
            'mixup': AUG_MIXUP,
        }

        # Merge additional kwargs
        train_args.update(kwargs)

        print(f"\nStarting training...")
        print(f"Results will be saved to: {self.results_dir / name}")

        # Train
        results = yolo_model.train(**train_args)

        print("\n" + "="*80)
        print("STAGE 1 TRAINING COMPLETE")
        print("="*80)
        print(f"\nBest weights: {self.results_dir / name / 'weights' / 'best.pt'}")
        print(f"Last weights: {self.results_dir / name / 'weights' / 'last.pt'}")
        print(f"Results plot: {self.results_dir / name / 'results.png'}")

        return results

    def train_stage2(
        self,
        weights: str,
        data: str = DEFAULT_STAGE2_DATA,
        epochs: int = DEFAULT_EPOCHS_STAGE2,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        batch: int = DEFAULT_BATCH_SIZE,
        device: str = DEFAULT_GPU_DEVICES,
        patience: int = DEFAULT_PATIENCE_STAGE2,
        workers: int = DEFAULT_NUM_WORKERS,
        cache: Union[bool, str] = DEFAULT_CACHE_ENABLED,
        optimizer: str = DEFAULT_OPTIMIZER,
        lr0: float = DEFAULT_LEARNING_RATE_STAGE2,
        name: str = 'yolov8m_pose_640_stage2',
        **kwargs
    ) -> Dict[str, Any]:
        """Train Stage 2: Fine-tuning with mixed data.

        Args:
            weights: Path to pretrained weights from Stage 1
            data: Mixed dataset config (synthetic + manual labels)
            epochs: Maximum fine-tuning epochs
            imgsz: Input image size
            batch: Batch size (total across all GPUs)
            device: GPU device(s)
            patience: Early stopping patience
            workers: Number of dataloader workers
            cache: Cache images in RAM/disk for faster training (True, False, or 'disk')
            optimizer: Optimizer type (default: AdamW)
            lr0: Initial learning rate for fine-tuning (default: 0.0005 for AdamW)
            name: Experiment name
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        print("\n" + "="*80)
        print("STAGE 2: FINE-TUNING WITH MIXED DATA")
        print("="*80)

        # Resolve paths
        weights_path = Path(weights)
        if not weights_path.exists():
            # Try relative to results_dir
            weights_path = self.results_dir / weights
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Weights not found: {weights}\n"
                    f"Also tried: {weights_path}"
                )

        data_path = self.configs_dir / data
        if not data_path.exists():
            print(f"\nWARNING: Dataset config not found: {data_path}")
            print(f"This is expected if manual labels haven't been collected yet.")
            print(f"Skipping Stage 2 fine-tuning.")
            return {}

        print(f"\nConfiguration:")
        print(f"  Pretrained weights: {weights_path}")
        print(f"  Data: {data_path}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Learning rate: {lr0}")
        print(f"  Device(s): {device}")

        # Load pretrained model
        print(f"\nLoading pretrained weights: {weights_path}")
        yolo_model = YOLO(str(weights_path))

        # Fine-tuning arguments
        train_args = {
            'data': str(data_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'workers': workers,
            'cache': cache,
            'project': str(self.results_dir),
            'name': name,
            'patience': patience,
            'save': True,
            'save_period': CHECKPOINT_SAVE_PERIOD,
            'verbose': True,
            'plots': True,
            # Optimizer configuration
            'optimizer': optimizer,
            'lr0': lr0,
            'cos_lr': True,  # Cosine learning rate scheduler
            'amp': True,  # Automatic Mixed Precision
            # Pose-specific loss weights
            'pose': 12.0,
            'kobj': 2.0,
        }

        # Merge additional kwargs
        train_args.update(kwargs)

        print(f"\nStarting fine-tuning...")
        results = yolo_model.train(**train_args)

        print("\n" + "="*80)
        print("STAGE 2 FINE-TUNING COMPLETE")
        print("="*80)
        print(f"\nBest weights: {self.results_dir / name / 'weights' / 'best.pt'}")

        return results

    def validate(
        self,
        weights: str,
        data: Optional[str] = None,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        batch: int = DEFAULT_BATCH_SIZE,
        device: str = '0',
        **kwargs
    ) -> Dict[str, Any]:
        """Validate trained model.

        Args:
            weights: Path to model weights
            data: Dataset config (uses same as training if None)
            imgsz: Input image size
            batch: Batch size for validation
            device: GPU device
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics dictionary
        """
        print("\n" + "="*80)
        print("MODEL VALIDATION")
        print("="*80)

        # Resolve weights path
        weights_path = Path(weights)
        if not weights_path.exists():
            weights_path = self.results_dir / weights
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights}")

        print(f"\nLoading weights: {weights_path}")
        yolo_model = YOLO(str(weights_path))

        # Validation arguments
        val_args = {
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'verbose': True,
        }

        if data is not None:
            data_path = self.configs_dir / data
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset config not found: {data_path}")
            val_args['data'] = str(data_path)

        val_args.update(kwargs)

        print(f"\nRunning validation...")
        results = yolo_model.val(**val_args)

        # Print key metrics
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)

        metrics = results.results_dict

        # Print all available metric keys for debugging
        print("\nAvailable metric keys:")
        for key in sorted(metrics.keys()):
            if isinstance(metrics[key], (int, float)):
                print(f"  {key}: {metrics[key]:.4f}")

        # Print key metrics (try common formats for box and keypoints)
        print(f"\nBox Metrics:")
        box_map50 = metrics.get('metrics/mAP50(B)') or metrics.get('box/mAP50') or 0.0
        box_map50_95 = metrics.get('metrics/mAP50-95(B)') or metrics.get('box/mAP50-95') or 0.0
        print(f"  mAP@50:    {box_map50:.4f}")
        print(f"  mAP@50-95: {box_map50_95:.4f}")

        print(f"\nKeypoint Metrics (OKS - Object Keypoint Similarity):")
        pose_map50 = (metrics.get('metrics/mAP50(P)') or
                      metrics.get('pose/mAP50') or
                      metrics.get('keypoints/mAP50') or 0.0)
        pose_map50_95 = (metrics.get('metrics/mAP50-95(P)') or
                         metrics.get('pose/mAP50-95') or
                         metrics.get('keypoints/mAP50-95') or 0.0)
        print(f"  mAP@50:    {pose_map50:.4f}")
        print(f"  mAP@50-95: {pose_map50_95:.4f}")

        print(f"\nTarget: OKS mAP@50 > 0.85")
        print("\nNOTE: If metrics show 0.0, check 'Available metric keys' above")
        print("      and update code with correct key names.")

        return metrics

    def export(
        self,
        weights: str,
        formats: Optional[List[str]] = None,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        half: bool = True,
        int8: bool = False,
        **kwargs
    ) -> Dict[str, Path]:
        """Export model for deployment.

        Args:
            weights: Path to model weights
            formats: Export formats (e.g., 'tflite' for mobile deployment)
            imgsz: Input image size
            half: Use FP16 precision (TFLite only)
            int8: Use INT8 quantization (TFLite only, requires calibration)
            **kwargs: Additional export arguments

        Returns:
            Dictionary mapping format to exported file path
        """
        if formats is None:
            formats = ['tflite']

        print("\n" + "="*80)
        print("MODEL EXPORT")
        print("="*80)

        # Resolve weights path
        weights_path = Path(weights)
        if not weights_path.exists():
            weights_path = self.results_dir / weights
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights}")

        print(f"\nLoading weights: {weights_path}")
        yolo_model = YOLO(str(weights_path))

        exported_files = {}

        for fmt in formats:
            print(f"\n{'='*40}")
            print(f"Exporting to {fmt.upper()}")
            print('='*40)

            export_args = {
                'format': fmt,
                'imgsz': imgsz,
            }

            # TFLite-specific arguments
            if fmt == 'tflite':
                if int8:
                    export_args['int8'] = True
                    print(f"  Quantization: INT8")
                else:
                    export_args['half'] = half
                    print(f"  Precision: FP16" if half else "  Precision: FP32")

            export_args.update(kwargs)

            # Export
            export_path = yolo_model.export(**export_args)
            exported_files[fmt] = Path(export_path)

            print(f"\nExported: {export_path}")

            # Usage instructions
            if fmt == 'tflite':
                print(f"\nAndroid Usage:")
                print(f"  Copy to: android/app/src/main/assets/")

        print("\n" + "="*80)
        print("EXPORT COMPLETE")
        print("="*80)

        return exported_files


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Training Pipeline for microPAD Auto-Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ZERO-CONFIG TRAINING (recommended - uses medium preset)
  python train_yolo.py

  # PRESET SELECTION (mutually exclusive)
  python train_yolo.py --medium  # 640px, high accuracy (DEFAULT)
  python train_yolo.py --small   # 640px, balanced speed/accuracy
  python train_yolo.py --nano    # 640px, fast inference

  # OVERRIDE RESOLUTION OR BATCH SIZE
  python train_yolo.py --medium --batch 48
  python train_yolo.py --small --batch 64

  # CUSTOM MODEL (auto-generates experiment name)
  python train_yolo.py --model yolov8n-pose.pt --imgsz 640 --batch 48

  # STAGE 2: Fine-tuning with manual labels
  python train_yolo.py --stage 2 --weights training_runs/yolov8m_pose_640/weights/best.pt

  # VALIDATE trained model
  python train_yolo.py --validate --weights training_runs/yolov8m_pose_640/weights/best.pt

  # EXPORT to TFLite for mobile deployment
  python train_yolo.py --export --weights training_runs/yolov8n_pose_640/weights/best.pt --imgsz 640

  # Export with FP32 precision (instead of default FP16)
  python train_yolo.py --export --weights training_runs/yolov8m_pose_640/weights/best.pt --no-half

  # Export with INT8 quantization
  python train_yolo.py --export --weights training_runs/yolov8m_pose_640/weights/best.pt --int8

3-Tier Preset System:
  | Preset    | Model          | Resolution | Batch | Per GPU | Use Case                |
  |-----------|----------------|------------|-------|---------|-------------------------|
  | medium    | yolov8m-pose   | 640px      | 32    | 16      | DEFAULT - High accuracy |
  | small     | yolov8s-pose   | 640px      | 48    | 24      | Balanced speed/accuracy |
  | nano      | yolov8n-pose   | 640px      | 64    | 32      | Fast training/inference |

Common Settings:
  Optimizer: AdamW
  Learning rate: 0.001 (stage 1), 0.0005 (stage 2)
  Cosine LR scheduler: Enabled
  Mixed precision (AMP): Enabled
  Pose loss weights: pose=12.0, kobj=2.0
  GPUs: 0,1 (dual A6000, NVLink interconnect)

Override Examples:
  python train_yolo.py --medium --batch 48   # medium with custom batch
  python train_yolo.py --small --batch 64    # small model with custom batch
        """
    )

    # Mode selection (defaults to Stage 1 training)
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--stage', type=int, choices=[1, 2], default=1,
                           help='Training stage (default: 1 - synthetic pretraining, 2: mixed fine-tuning)')
    mode_group.add_argument('--validate', action='store_true',
                           help='Validate trained model')
    mode_group.add_argument('--export', action='store_true',
                           help='Export model for deployment')

    # Preset selection (mutually exclusive)
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument('--medium', action='store_true',
                       help='Medium preset: yolov8m @ 640px (DEFAULT)')
    preset_group.add_argument('--small', action='store_true',
                       help='Small preset: yolov8s @ 640px')
    preset_group.add_argument('--nano', action='store_true',
                       help='Nano preset: yolov8n @ 640px')

    # Common arguments
    parser.add_argument('--weights', type=str,
                       help='Path to model weights (required for stage 2, validate, export)')
    parser.add_argument('--model', type=str, default=None,
                       help=f'Base model to train (default: auto from preset)')
    parser.add_argument('--device', type=str, default=DEFAULT_GPU_DEVICES,
                       help=f'GPU device(s) (default: {DEFAULT_GPU_DEVICES})')
    parser.add_argument('--imgsz', type=int, default=None,
                       help=f'Input image size (default: auto from preset)')
    parser.add_argument('--name', type=str,
                       help='Experiment name for results directory (default: auto-generated based on preset)')

    # Training arguments
    parser.add_argument('--epochs', type=int,
                       help=f'Training epochs (default: {DEFAULT_EPOCHS_STAGE1} for stage 1, {DEFAULT_EPOCHS_STAGE2} for stage 2)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (default: auto from preset)')
    parser.add_argument('--patience', type=int,
                       help=f'Early stopping patience (default: {DEFAULT_PATIENCE_STAGE1} for stage 1, {DEFAULT_PATIENCE_STAGE2} for stage 2)')
    parser.add_argument('--lr0', type=float,
                       help=f'Initial learning rate (default: {DEFAULT_LEARNING_RATE_STAGE1} for stage 1, {DEFAULT_LEARNING_RATE_STAGE2} for stage 2)')

    # Advanced training arguments
    parser.add_argument('--workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help=f'Number of dataloader workers (default: {DEFAULT_NUM_WORKERS})')
    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE_ENABLED, choices=['True', 'False', 'disk'],
                       help=f'Image caching: True (RAM), False (disabled), disk (disk cache) (default: {DEFAULT_CACHE_ENABLED})')
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER, choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
                       help=f'Optimizer (default: {DEFAULT_OPTIMIZER})')
    parser.add_argument('--cos-lr', action='store_true', default=True,
                       help='Enable cosine learning rate scheduler (default: enabled)')

    # Export arguments
    parser.add_argument('--formats', nargs='+', default=['tflite'],
                       choices=['tflite', 'torchscript', 'coreml'],
                       help='Export formats (default: tflite for Android)')
    export_precision = parser.add_mutually_exclusive_group()
    export_precision.add_argument('--half', dest='half', action='store_true',
                       help='Use FP16 precision for TFLite export (default)')
    export_precision.add_argument('--no-half', dest='half', action='store_false',
                       help='Use FP32 precision for TFLite export')
    parser.set_defaults(half=True)
    parser.add_argument('--int8', action='store_true',
                       help='Use INT8 quantization for TFLite')

    # Augmentation arguments
    parser.add_argument('--mixup', type=float, default=AUG_MIXUP,
                       help=f'MixUp augmentation probability (default: {AUG_MIXUP})')

    # Data arguments
    parser.add_argument('--data', type=str,
                       help='Dataset config (default: micropad_synth.yaml for stage 1, micropad_mixed.yaml for stage 2)')

    args = parser.parse_args()

    # Initialize trainer
    try:
        trainer = YOLOTrainer()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Determine which preset to use (default: medium)
    if args.small:
        preset_name = 'small'
        preset_model = PRESET_SMALL_MODEL
        preset_imgsz = PRESET_SMALL_IMGSZ
        preset_batch = PRESET_SMALL_BATCH
        preset_default_name = DEFAULT_NAME_SMALL
    elif args.nano:
        preset_name = 'nano'
        preset_model = PRESET_NANO_MODEL
        preset_imgsz = PRESET_NANO_IMGSZ
        preset_batch = PRESET_NANO_BATCH
        preset_default_name = DEFAULT_NAME_NANO
    else:
        # Default: medium preset (includes explicit --medium)
        preset_name = 'medium'
        preset_model = PRESET_MEDIUM_MODEL
        preset_imgsz = PRESET_MEDIUM_IMGSZ
        preset_batch = PRESET_MEDIUM_BATCH
        preset_default_name = DEFAULT_NAME_MEDIUM

    # Apply preset defaults (user overrides take precedence)
    if args.model is None:
        args.model = preset_model
    if args.imgsz is None:
        args.imgsz = preset_imgsz
    if args.batch is None:
        args.batch = preset_batch
    if args.name is None:
        args.name = preset_default_name

    # Print preset configuration
    print(f"\n[{preset_name.upper()} Preset] Configuration:")
    print(f"  Model: {args.model}" + (" (override)" if args.model != preset_model else ""))
    print(f"  Image size: {args.imgsz}" + (" (override)" if args.imgsz != preset_imgsz else ""))
    print(f"  Batch size: {args.batch}" + (" (override)" if args.batch != preset_batch else ""))
    if preset_name == 'nano':
        print(f"  Augmentation: scale={AUG_SCALE_MOBILE}, translate={AUG_TRANSLATE_MOBILE}, mosaic={AUG_MOSAIC_MOBILE}")

    # Execute requested mode
    try:
        if args.stage == 1:
            # Stage 1: Synthetic pretraining
            train_kwargs = {}
            train_kwargs['model'] = args.model  # Pass model to trainer
            if args.epochs:
                train_kwargs['epochs'] = args.epochs
            if args.batch:
                train_kwargs['batch'] = args.batch
            if args.patience:
                train_kwargs['patience'] = args.patience
            if args.lr0:
                train_kwargs['lr0'] = args.lr0
            if args.data:
                train_kwargs['data'] = args.data

            # Nano preset augmentation overrides (smaller features)
            if preset_name == 'nano':
                train_kwargs['scale'] = AUG_SCALE_MOBILE
                train_kwargs['translate'] = AUG_TRANSLATE_MOBILE
                train_kwargs['mosaic'] = AUG_MOSAIC_MOBILE

            # MixUp augmentation (passed through to train_args)
            if args.mixup > 0:
                train_kwargs['mixup'] = args.mixup

            # Advanced options
            train_kwargs['workers'] = args.workers
            train_kwargs['optimizer'] = args.optimizer
            # Convert cache string to appropriate type
            if args.cache == 'True':
                train_kwargs['cache'] = True
            elif args.cache == 'False':
                train_kwargs['cache'] = False
            else:
                train_kwargs['cache'] = args.cache  # 'disk'

            # cos_lr is already enabled by default in train_args, but can be overridden
            if not args.cos_lr:
                train_kwargs['cos_lr'] = False

            trainer.train_stage1(
                device=args.device,
                imgsz=args.imgsz,
                name=args.name,
                **train_kwargs
            )

        elif args.stage == 2:
            # Stage 2: Fine-tuning
            if not args.weights:
                print("ERROR: --weights required for Stage 2 fine-tuning")
                print(f"Example: --weights training_runs/yolov8m_pose_640/weights/best.pt")
                sys.exit(1)

            train_kwargs = {}
            if args.epochs:
                train_kwargs['epochs'] = args.epochs
            if args.batch:
                train_kwargs['batch'] = args.batch
            if args.patience:
                train_kwargs['patience'] = args.patience
            if args.lr0:
                train_kwargs['lr0'] = args.lr0
            if args.data:
                train_kwargs['data'] = args.data

            # Nano preset augmentation overrides
            if preset_name == 'nano':
                train_kwargs['scale'] = AUG_SCALE_MOBILE
                train_kwargs['translate'] = AUG_TRANSLATE_MOBILE
                train_kwargs['mosaic'] = AUG_MOSAIC_MOBILE

            # MixUp augmentation (passed through to train_args)
            if args.mixup > 0:
                train_kwargs['mixup'] = args.mixup

            # Advanced options
            train_kwargs['workers'] = args.workers
            train_kwargs['optimizer'] = args.optimizer
            # Convert cache string to appropriate type
            if args.cache == 'True':
                train_kwargs['cache'] = True
            elif args.cache == 'False':
                train_kwargs['cache'] = False
            else:
                train_kwargs['cache'] = args.cache  # 'disk'

            # cos_lr is already enabled by default in train_args, but can be overridden
            if not args.cos_lr:
                train_kwargs['cos_lr'] = False

            # Append _stage2 to name for fine-tuning runs
            stage2_name = f"{args.name}_stage2" if not args.name.endswith('_stage2') else args.name
            trainer.train_stage2(
                weights=args.weights,
                device=args.device,
                imgsz=args.imgsz,
                name=stage2_name,
                **train_kwargs
            )

        elif args.validate:
            # Validation
            if not args.weights:
                print("ERROR: --weights required for validation")
                print(f"Example: --weights training_runs/yolov8m_pose_640/weights/best.pt")
                sys.exit(1)

            trainer.validate(
                weights=args.weights,
                data=args.data,
                imgsz=args.imgsz,
                device=args.device
            )

        elif args.export:
            # Export
            if not args.weights:
                print("ERROR: --weights required for export")
                print(f"Example: --weights training_runs/yolov8m_pose_640/weights/best.pt")
                sys.exit(1)

            trainer.export(
                weights=args.weights,
                formats=args.formats,
                imgsz=args.imgsz,
                half=args.half,
                int8=args.int8
            )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
