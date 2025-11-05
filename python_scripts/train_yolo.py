#!/usr/bin/env python3
"""
YOLOv11-pose Training Script for microPAD Quadrilateral Corner Detection

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

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
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
# These defaults are optimized for dual RTX A6000 workstation (96 GB VRAM total).
# Simply run: python train_yolo.py
# No command-line arguments required for optimal performance!
#
# To customize: Override any parameter via CLI flags (see --help)
# ============================================================================

# Model configuration
DEFAULT_MODEL = 'yolo11m-pose.pt'
DEFAULT_IMAGE_SIZE = 960

# Training hyperparameters (optimized for dual A6000 workstation)
DEFAULT_BATCH_SIZE = 32              # 16 per GPU on dual A6000 setup
DEFAULT_EPOCHS_STAGE1 = 150          # Sufficient for convergence on synthetic data
DEFAULT_EPOCHS_STAGE2 = 80           # Fine-tuning with early stopping
DEFAULT_PATIENCE_STAGE1 = 20         # Early stopping patience
DEFAULT_PATIENCE_STAGE2 = 15
DEFAULT_LEARNING_RATE_STAGE2 = 0.01  # Lower LR for fine-tuning

# Hardware configuration (optimized for 64-core CPU + dual RTX A6000)
# GPU device selection: Use CUDA_VISIBLE_DEVICES environment variable or default
# Default '0,2' selects dual RTX A6000 GPUs (skips RTX 3090 for homogeneous pairing)
DEFAULT_GPU_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0,2')
DEFAULT_NUM_WORKERS = 32             # Half of 64 CPU cores for data loading
DEFAULT_CACHE_ENABLED = 'disk'       # Disk cache for deterministic training

# Checkpoint configuration
CHECKPOINT_SAVE_PERIOD = 10          # Save checkpoint every N epochs

# Augmentation configuration
AUG_HSV_HUE = 0.015
AUG_HSV_SATURATION = 0.7
AUG_HSV_VALUE = 0.4
AUG_TRANSLATE = 0.1
AUG_SCALE = 0.5
AUG_FLIP_LR = 0.5
AUG_MOSAIC = 1.0
AUG_ROTATION = 0.0  # Disabled (already in synthetic data)

# Dataset configuration
DEFAULT_STAGE1_DATA = 'micropad_synth.yaml'
DEFAULT_STAGE2_DATA = 'micropad_mixed.yaml'

# ============================================================================

# Verify Ultralytics version supports pose training
def check_ultralytics_version():
    """Check that Ultralytics version supports YOLOv11-pose."""
    required_version = (8, 1, 0)
    try:
        version_parts = tuple(int(x) for x in ultralytics_version.split('.')[:3])
        if version_parts < required_version:
            print(f"WARNING: Ultralytics version {ultralytics_version} may not fully support YOLOv11-pose.")
            print(f"         Recommended version: {'.'.join(map(str, required_version))} or higher")
            print(f"         Update with: pip install --upgrade ultralytics")
    except Exception:
        print(f"WARNING: Could not parse Ultralytics version: {ultralytics_version}")

check_ultralytics_version()


class YOLOTrainer:
    """YOLOv11 training pipeline for microPAD auto-detection."""

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
        name: str = 'yolo11m_pose',
        **kwargs
    ) -> Dict[str, Any]:
        """Train Stage 1: Synthetic data pretraining.

        Args:
            model: YOLOv11 pretrained model for keypoint detection
            data: Dataset config file in configs/ directory
            epochs: Maximum training epochs
            imgsz: Input image size
            batch: Batch size (total across all GPUs)
            device: GPU device(s) (e.g., '0' for single GPU, '0,1' for multi-GPU)
            patience: Early stopping patience
            workers: Number of dataloader workers
            cache: Cache images in RAM/disk for faster training (True, False, or 'disk')
            name: Experiment name
            **kwargs: Additional training arguments passed to YOLO

        Returns:
            Training results dictionary
        """
        print("\n" + "="*80)
        print("STAGE 1: SYNTHETIC DATA PRETRAINING")
        print("="*80)

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
            # Augmentation configuration
            'hsv_h': AUG_HSV_HUE,
            'hsv_s': AUG_HSV_SATURATION,
            'hsv_v': AUG_HSV_VALUE,
            'translate': AUG_TRANSLATE,
            'scale': AUG_SCALE,
            'fliplr': AUG_FLIP_LR,
            'mosaic': AUG_MOSAIC,
            'degrees': AUG_ROTATION,
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
        lr0: float = DEFAULT_LEARNING_RATE_STAGE2,
        name: str = 'yolo11m_pose_stage2',
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
            lr0: Initial learning rate for fine-tuning
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
            'lr0': lr0,
            'save': True,
            'save_period': CHECKPOINT_SAVE_PERIOD,
            'verbose': True,
            'plots': True,
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
        formats: list = ['tflite'],
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='YOLOv11 Training Pipeline for microPAD Auto-Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Stage 1 (synthetic data pretraining) - default mode
  python train_yolo.py
  python train_yolo.py --stage 1  # Explicit stage 1

  # Train Stage 2 (fine-tuning with manual labels)
  python train_yolo.py --stage 2 --weights training_runs/yolo11m_pose/weights/best.pt

  # Validate model
  python train_yolo.py --validate --weights training_runs/yolo11m_pose/weights/best.pt

  # Export to TFLite for mobile deployment
  python train_yolo.py --export --weights training_runs/yolo11m_pose/weights/best.pt

  # Export with FP32 precision (instead of default FP16)
  python train_yolo.py --export --weights training_runs/yolo11m_pose/weights/best.pt --no-half

  # Export with INT8 quantization
  python train_yolo.py --export --weights training_runs/yolo11m_pose/weights/best.pt --int8

  # Custom training parameters (adjust batch size, workers, cache strategy)
  python train_yolo.py --stage 1 --epochs 200 --batch 128 --workers 16 --cache disk

  # Advanced: custom optimizer and learning rate scheduler
  python train_yolo.py --stage 1 --optimizer AdamW --cos-lr
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

    # Common arguments
    parser.add_argument('--weights', type=str,
                       help='Path to model weights (required for stage 2, validate, export)')
    parser.add_argument('--device', type=str, default=DEFAULT_GPU_DEVICES,
                       help=f'GPU device(s) (default: {DEFAULT_GPU_DEVICES})')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMAGE_SIZE,
                       help=f'Input image size (default: {DEFAULT_IMAGE_SIZE})')

    # Training arguments
    parser.add_argument('--epochs', type=int,
                       help=f'Training epochs (default: {DEFAULT_EPOCHS_STAGE1} for stage 1, {DEFAULT_EPOCHS_STAGE2} for stage 2)')
    parser.add_argument('--batch', type=int,
                       help=f'Batch size (default: {DEFAULT_BATCH_SIZE}, distributed across GPUs)')
    parser.add_argument('--patience', type=int,
                       help=f'Early stopping patience (default: {DEFAULT_PATIENCE_STAGE1} for stage 1, {DEFAULT_PATIENCE_STAGE2} for stage 2)')
    parser.add_argument('--lr0', type=float,
                       help=f'Initial learning rate (default: {DEFAULT_LEARNING_RATE_STAGE2} for stage 2)')

    # Advanced training arguments
    parser.add_argument('--workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help=f'Number of dataloader workers (default: {DEFAULT_NUM_WORKERS})')
    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE_ENABLED, choices=['True', 'False', 'disk'],
                       help=f'Image caching: True (RAM), False (disabled), disk (disk cache) (default: {DEFAULT_CACHE_ENABLED})')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
                       help='Optimizer (default: auto - lets YOLO choose)')
    parser.add_argument('--cos-lr', action='store_true',
                       help='Enable cosine learning rate scheduler')

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

    # Execute requested mode
    try:
        if args.stage == 1:
            # Stage 1: Synthetic pretraining
            train_kwargs = {}
            if args.epochs:
                train_kwargs['epochs'] = args.epochs
            if args.batch:
                train_kwargs['batch'] = args.batch
            if args.patience:
                train_kwargs['patience'] = args.patience
            if args.data:
                train_kwargs['data'] = args.data

            # Advanced options
            train_kwargs['workers'] = args.workers
            # Convert cache string to appropriate type
            if args.cache == 'True':
                train_kwargs['cache'] = True
            elif args.cache == 'False':
                train_kwargs['cache'] = False
            else:
                train_kwargs['cache'] = args.cache  # 'disk'

            if args.optimizer:
                train_kwargs['optimizer'] = args.optimizer
            if args.cos_lr:
                train_kwargs['cos_lr'] = True

            trainer.train_stage1(
                device=args.device,
                imgsz=args.imgsz,
                **train_kwargs
            )

        elif args.stage == 2:
            # Stage 2: Fine-tuning
            if not args.weights:
                print("ERROR: --weights required for Stage 2 fine-tuning")
                print("Example: --weights training_runs/yolo11m_pose/weights/best.pt")
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

            # Advanced options
            train_kwargs['workers'] = args.workers
            # Convert cache string to appropriate type
            if args.cache == 'True':
                train_kwargs['cache'] = True
            elif args.cache == 'False':
                train_kwargs['cache'] = False
            else:
                train_kwargs['cache'] = args.cache  # 'disk'

            if args.optimizer:
                train_kwargs['optimizer'] = args.optimizer
            if args.cos_lr:
                train_kwargs['cos_lr'] = True

            trainer.train_stage2(
                weights=args.weights,
                device=args.device,
                imgsz=args.imgsz,
                **train_kwargs
            )

        elif args.validate:
            # Validation
            if not args.weights:
                print("ERROR: --weights required for validation")
                print("Example: --weights training_runs/yolo11m_pose/weights/best.pt")
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
                print("Example: --weights training_runs/yolo11m_pose/weights/best.pt")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
