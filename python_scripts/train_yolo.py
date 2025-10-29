#!/usr/bin/env python3
"""
YOLOv11 Training Script for microPAD Quadrilateral Auto-Detection

QUICK START (VS Code Terminal):
    conda activate microPAD-python-env
    python train_yolo.py --stage 1

This will start Stage 1 training with optimized defaults:
- Devices: GPUs 0,1 (dual A6000 with NVLink)
- Batch size: 32 (optimized for 126-image training set)
- Epochs: 150 with early stopping (patience=20)
- Results: micropad_detection/yolo11n_synth/

TRAINING PIPELINE:
- Stage 1: Pretraining on synthetic data (current)
- Stage 2: Fine-tuning on mixed synthetic + manual labels (future)

Also provides validation and export capabilities for MATLAB/Android deployment.

HARDWARE REQUIREMENTS:
- GPUs: Dual RTX A6000 (48GB each, NVLink connected on CUDA devices 0,1)
- RAM: 256GB
- CPUs: 64 cores / 128 threads
- Storage: 20TB NVMe

PERFORMANCE TARGETS:
- Mask mAP@50 > 0.85
- Detection rate > 85%
- Inference < 50ms on Android devices
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

try:
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings
except ImportError:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


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
        self.results_dir = self.project_root / 'micropad_detection'

        # Verify project structure
        if not self.configs_dir.exists():
            raise FileNotFoundError(
                f"Configs directory not found: {self.configs_dir}\n"
                f"Expected structure: {self.project_root}/python_scripts/configs/"
            )

    def train_stage1(
        self,
        model: str = 'yolo11n-seg.pt',
        data: str = 'micropad_synth.yaml',
        epochs: int = 150,
        imgsz: int = 640,
        batch: int = 32,
        device: str = '0,1',
        patience: int = 20,
        name: str = 'yolo11n_synth',
        **kwargs
    ) -> Dict[str, Any]:
        """Train Stage 1: Synthetic data pretraining.

        Args:
            model: YOLOv11 pretrained model (yolo11n-seg.pt recommended)
            data: Dataset config file in configs/ directory
            epochs: Maximum training epochs
            imgsz: Input image size (640 recommended)
            batch: Batch size (32 optimized for dataset size)
            device: GPU device(s) - '0,1' for dual A6000 GPUs, '0' for single
            patience: Early stopping patience
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

        print(f"\nConfiguration:")
        print(f"  Model: {model}")
        print(f"  Data: {data_path}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
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
            'project': str(self.results_dir),
            'name': name,
            'patience': patience,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'verbose': True,
            'plots': True,
            # Augmentation (no rotation - already in synthetic data)
            'hsv_h': 0.015,      # HSV hue augmentation
            'hsv_s': 0.7,        # HSV saturation augmentation
            'hsv_v': 0.4,        # HSV value augmentation
            'translate': 0.1,    # Translation augmentation
            'scale': 0.5,        # Scale augmentation
            'fliplr': 0.5,       # Horizontal flip probability
            'mosaic': 1.0,       # Mosaic augmentation probability
            'degrees': 0.0,      # No rotation (already in synthetic)
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
        data: str = 'micropad_mixed.yaml',
        epochs: int = 80,
        imgsz: int = 640,
        batch: int = 96,
        device: str = '0,1',
        patience: int = 15,
        lr0: float = 0.01,
        name: str = 'yolo11n_mixed',
        **kwargs
    ) -> Dict[str, Any]:
        """Train Stage 2: Fine-tuning with mixed data.

        Args:
            weights: Path to pretrained weights from Stage 1
            data: Mixed dataset config (synthetic + manual labels)
            epochs: Maximum fine-tuning epochs (less than Stage 1)
            imgsz: Input image size
            batch: Batch size (smaller for fine-tuning)
            device: GPU device(s)
            patience: Early stopping patience
            lr0: Initial learning rate (lower for fine-tuning)
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
            'project': str(self.results_dir),
            'name': name,
            'patience': patience,
            'lr0': lr0,
            'save': True,
            'save_period': 10,
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
        imgsz: int = 640,
        batch: int = 32,
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

        # Print key metrics (try common formats)
        print(f"\nBox Metrics:")
        box_map50 = metrics.get('metrics/mAP50(B)') or metrics.get('box/mAP50') or 0.0
        box_map50_95 = metrics.get('metrics/mAP50-95(B)') or metrics.get('box/mAP50-95') or 0.0
        print(f"  mAP@50:    {box_map50:.4f}")
        print(f"  mAP@50-95: {box_map50_95:.4f}")

        print(f"\nMask Metrics:")
        mask_map50 = metrics.get('metrics/mAP50(M)') or metrics.get('mask/mAP50') or 0.0
        mask_map50_95 = metrics.get('metrics/mAP50-95(M)') or metrics.get('mask/mAP50-95') or 0.0
        print(f"  mAP@50:    {mask_map50:.4f}")
        print(f"  mAP@50-95: {mask_map50_95:.4f}")

        print(f"\nTarget: Mask mAP@50 > 0.85")
        print("\nNOTE: If metrics show 0.0, check 'Available metric keys' above")
        print("      and update code with correct key names.")

        return metrics

    def export(
        self,
        weights: str,
        formats: list = ['onnx', 'tflite'],
        imgsz: int = 640,
        half: bool = True,
        simplify: bool = True,
        int8: bool = False,
        **kwargs
    ) -> Dict[str, Path]:
        """Export model for deployment.

        Args:
            weights: Path to model weights
            formats: Export formats ('onnx', 'tflite', etc.)
            imgsz: Input image size
            half: Use FP16 precision (TFLite only)
            simplify: Simplify ONNX model
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

            # Format-specific arguments
            if fmt == 'onnx':
                export_args['simplify'] = simplify
                print(f"  Simplify: {simplify}")
            elif fmt == 'tflite':
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
            if fmt == 'onnx':
                print(f"\nMATLAB Usage:")
                print(f"  net = importONNXNetwork('{export_path}');")
            elif fmt == 'tflite':
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
  # Train Stage 1 (synthetic data pretraining)
  python train_yolo.py --stage 1

  # Train Stage 2 (fine-tuning with manual labels)
  python train_yolo.py --stage 2 --weights micropad_detection/yolo11n_synth/weights/best.pt

  # Validate model
  python train_yolo.py --validate --weights micropad_detection/yolo11n_synth/weights/best.pt

  # Export to ONNX and TFLite
  python train_yolo.py --export --weights micropad_detection/yolo11n_synth/weights/best.pt

  # Export with INT8 quantization for Android
  python train_yolo.py --export --weights best.pt --formats tflite --int8

  # Custom training parameters
  python train_yolo.py --stage 1 --epochs 200 --batch 96 --device 0
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--stage', type=int, choices=[1, 2],
                           help='Training stage (1: synthetic, 2: mixed)')
    mode_group.add_argument('--validate', action='store_true',
                           help='Validate trained model')
    mode_group.add_argument('--export', action='store_true',
                           help='Export model for deployment')

    # Common arguments
    parser.add_argument('--weights', type=str,
                       help='Path to model weights (required for stage 2, validate, export)')
    parser.add_argument('--device', type=str, default='0,1',
                       help='GPU device(s) (default: 0,1 for NVLink A6000 GPUs)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')

    # Training arguments
    parser.add_argument('--epochs', type=int,
                       help='Training epochs (default: 150 for stage 1, 80 for stage 2)')
    parser.add_argument('--batch', type=int,
                       help='Batch size (default: 32 for stage 1 optimized for dataset size, 96 for stage 2)')
    parser.add_argument('--patience', type=int,
                       help='Early stopping patience (default: 20 for stage 1, 15 for stage 2)')
    parser.add_argument('--lr0', type=float,
                       help='Initial learning rate (default: 0.01 for stage 2)')

    # Export arguments
    parser.add_argument('--formats', nargs='+', default=['onnx', 'tflite'],
                       choices=['onnx', 'tflite', 'torchscript', 'coreml'],
                       help='Export formats (default: onnx tflite)')
    parser.add_argument('--half', action='store_true', default=True,
                       help='Use FP16 for TFLite (default: True)')
    parser.add_argument('--int8', action='store_true',
                       help='Use INT8 quantization for TFLite')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model (default: True)')

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

            trainer.train_stage1(
                device=args.device,
                imgsz=args.imgsz,
                **train_kwargs
            )

        elif args.stage == 2:
            # Stage 2: Fine-tuning
            if not args.weights:
                print("ERROR: --weights required for Stage 2 fine-tuning")
                print("Example: --weights micropad_detection/yolo11n_synth/weights/best.pt")
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
                print("Example: --weights micropad_detection/yolo11n_synth/weights/best.pt")
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
                print("Example: --weights micropad_detection/yolo11n_synth/weights/best.pt")
                sys.exit(1)

            trainer.export(
                weights=args.weights,
                formats=args.formats,
                imgsz=args.imgsz,
                half=args.half,
                simplify=args.simplify,
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
