"""
Prepare YOLO dataset configuration and train/val splits for microPAD auto-detection.

This script restructures the augmented dataset for YOLO training:
1. Generates YOLO labels from MATLAB coordinates (augmented_2_micropads/coordinates.txt)
2. Moves images from augmented_1_dataset/[phone]/ to augmented_1_dataset/[phone]/images/
3. Creates labels in augmented_1_dataset/[phone]/labels/
4. Creates train.txt and val.txt with absolute paths

MATLAB scripts (augment_dataset.m) generate images and polygon coordinates.
This Python script converts coordinates to YOLO pose keypoint format and restructures directories.

Label Format (YOLOv11-pose - DEFAULT):
    class_id x_center y_center width height x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    - Bounding box: axis-aligned bbox around keypoints (normalized [0,1])
    - Keypoints: 4 corners ordered clockwise from top-left (TL, TR, BR, BL)
    - Visibility flags: 2 = visible (all corners always visible in our dataset)
    - Default format is 'pose' for keypoint detection

Configuration:
    - Train phones: iphone_11, iphone_15, realme_c55
    - Val phone: samsung_a75

Usage:
    python prepare_yolo_dataset.py [--format pose|seg]
    Default: python prepare_yolo_dataset.py  # Uses pose format
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
import yaml
import cv2
import numpy as np

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
AUGMENTED_DATASET = PROJECT_ROOT / "augmented_1_dataset"
PHONE_DIRS = ["iphone_11", "iphone_15", "realme_c55", "samsung_a75"]
VAL_PHONE = "samsung_a75"  # Reserve one phone for validation
TRAIN_PHONES = [p for p in PHONE_DIRS if p != VAL_PHONE]

# Output paths
CONFIGS_DIR = PROJECT_ROOT / "python_scripts" / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)


def restructure_for_yolo() -> int:
    """Restructure dataset to YOLO-compatible format.

    Moves images from phone root to images/ subdirectory if not already there.
    Labels stay in labels/ subdirectory (MATLAB already puts them there).
    Function is idempotent - safe to run multiple times.

    Returns:
        Number of images moved
    """
    total_moved = 0

    for phone in PHONE_DIRS:
        phone_path = AUGMENTED_DATASET / phone
        images_dir = phone_path / "images"

        images_dir.mkdir(exist_ok=True)

        images_in_root = []
        for ext in ["*.png"]:
            for img_path in phone_path.glob(ext):
                if img_path.parent == phone_path:
                    images_in_root.append(img_path)

        if not images_in_root:
            continue

        moved_count = 0
        for img_path in images_in_root:
            dest_path = images_dir / img_path.name
            if not dest_path.exists():
                shutil.move(img_path, dest_path)
                moved_count += 1

        if moved_count > 0:
            print(f"  {phone}: moved {moved_count} images to images/")
            total_moved += moved_count

    return total_moved


def validate_clockwise_order(vertices: np.ndarray) -> bool:
    """Validate that vertices are ordered clockwise using signed area test.

    In image coordinates (Y-axis points down), "visually clockwise" ordering
    (TL→TR→BR→BL) produces positive signed area.

    Args:
        vertices: 4x2 numpy array of vertices

    Returns:
        True if clockwise in image coordinates, False otherwise
    """
    # Calculate signed area using shoelace formula
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]

    # In image coordinates (Y down), clockwise has positive signed area
    return area > 0


def order_keypoints_clockwise(vertices: np.ndarray) -> np.ndarray:
    """Order vertices clockwise starting from top-left.

    Top-left is defined as the vertex with minimum (x + y).
    Uses centroid-based angle sorting to establish clockwise order.

    Args:
        vertices: 4x2 numpy array of vertices (any order)

    Returns:
        4x2 numpy array ordered clockwise from top-left (TL, TR, BR, BL)
    """
    # Find centroid
    centroid = vertices.mean(axis=0)

    # Calculate angles from centroid to each vertex
    angles = np.arctan2(vertices[:, 1] - centroid[1],
                        vertices[:, 0] - centroid[0])

    # Sort vertices by angle (counter-clockwise from positive x-axis)
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    # Check if ordering is clockwise or counter-clockwise
    if not validate_clockwise_order(sorted_vertices):
        # Reverse to make clockwise (keep first vertex as anchor)
        sorted_vertices = sorted_vertices[::-1]

    # Find top-left vertex (minimum x + y)
    sum_coords = sorted_vertices[:, 0] + sorted_vertices[:, 1]
    top_left_idx = np.argmin(sum_coords)

    # Rotate array to start from top-left
    ordered_vertices = np.roll(sorted_vertices, -top_left_idx, axis=0)

    return ordered_vertices


def collect_image_paths(phone_dir: str, use_absolute_paths: bool = True) -> List[str]:
    """Collect all image paths from a phone directory.

    Args:
        phone_dir: Phone directory name (e.g., 'iphone_11')
        use_absolute_paths: If True, return absolute paths; if False, return relative paths

    Returns:
        Sorted list of image paths (absolute by default)

    Raises:
        FileNotFoundError: If images/ directory does not exist
        ValueError: If no images found in directory
    """
    phone_path = AUGMENTED_DATASET / phone_dir / "images"

    if not phone_path.exists():
        raise FileNotFoundError(f"Images directory not found: {phone_path}")

    images = []
    for ext in ["*.png"]:
        images.extend(phone_path.glob(ext))

    if not images:
        raise ValueError(f"No images found in {phone_path}")

    if use_absolute_paths:
        images = [str(img.absolute()) for img in images]
    else:
        images = [f"{phone_dir}/images/{img.name}" for img in images]

    return sorted(images)


def generate_yolo_labels(label_format: str = 'pose') -> Tuple[int, int]:
    """Generate YOLO labels from MATLAB coordinates.

    Reads polygon coordinates from augmented_2_micropads/[phone]/coordinates.txt
    and creates YOLOv11 labels in augmented_1_dataset/[phone]/labels/.

    Label formats:
        - 'pose': class_id x1 y1 2 x2 y2 2 x3 y3 2 x4 y4 2 (keypoint format, default)
        - 'seg': class_id x1 y1 x2 y2 x3 y3 x4 y4 (segmentation format, deprecated)

    Vertices are automatically ordered clockwise from top-left (TL, TR, BR, BL).

    Args:
        label_format: Label format to generate ('pose' or 'seg')

    Returns:
        Tuple of (total_labels_created, total_polygons_processed)
    """
    total_labels = 0
    total_polygons = 0

    micropads_dir = PROJECT_ROOT / "augmented_2_micropads"

    for phone in PHONE_DIRS:
        coord_file = micropads_dir / phone / "coordinates.txt"
        if not coord_file.exists():
            print(f"⚠️  Warning: coordinates.txt not found for {phone}")
            continue

        labels_dir = AUGMENTED_DATASET / phone / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Read coordinates (format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation)
        with open(coord_file, 'r') as f:
            lines = f.readlines()

        # Skip header line if present (case-insensitive, handle whitespace)
        if lines and lines[0].strip().lower().startswith('image'):
            lines = lines[1:]

        # Group polygons by image
        image_polygons = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 10:
                print(f"⚠️  Warning: malformed coordinate line (expected 10 fields, got {len(parts)}): {line.strip()[:80]}")
                continue

            img_name = parts[0]
            # Remove _con_N suffix to get base image name (handles names with underscores)
            # Format: {base_name}_con_{N} where base_name may contain underscores
            if '_con_' in img_name:
                # Find last occurrence of _con_ pattern
                idx = img_name.rfind('_con_')
                base_name = img_name[:idx]
            else:
                # Fallback: no _con_ suffix (shouldn't happen but safe)
                base_name = img_name

            # Extract polygon vertices (columns 2-9: x1 y1 x2 y2 x3 y3 x4 y4)
            vertices = [float(parts[i]) for i in range(2, 10)]
            polygon = [(vertices[i], vertices[i+1]) for i in range(0, 8, 2)]

            if base_name not in image_polygons:
                image_polygons[base_name] = []
            image_polygons[base_name].append(polygon)

        # Get image dimensions and create labels
        images_dir = AUGMENTED_DATASET / phone / "images"
        for base_name, polygons in image_polygons.items():
            # Find corresponding image file
            img_path = None
            for ext in ['.png']:
                candidate = images_dir / f"{base_name}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"⚠️  Warning: image not found for {base_name}")
                continue

            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Warning: failed to read {img_path}")
                continue
            height, width = img.shape[:2]

            # Validate coordinates are within image bounds
            valid_polygons = []
            for polygon in polygons:
                if all(0 <= x < width and 0 <= y < height for x, y in polygon):
                    valid_polygons.append(polygon)
                else:
                    print(f"⚠️  Warning: polygon outside image bounds in {base_name} (image: {width}x{height})")

            if not valid_polygons:
                print(f"⚠️  Warning: no valid polygons for {base_name}, skipping")
                continue

            # Write label file with only valid polygons
            label_path = labels_dir / f"{base_name}.txt"
            with open(label_path, 'w') as f:
                for polygon in valid_polygons:
                    # Convert to numpy array for ordering
                    vertices = np.array(polygon, dtype=np.float64)

                    # Order vertices clockwise from top-left
                    ordered_vertices = order_keypoints_clockwise(vertices)

                    # Normalize coordinates to [0, 1]
                    norm_vertices = ordered_vertices.copy()
                    norm_vertices[:, 0] /= width
                    norm_vertices[:, 1] /= height

                    # Verify ordering is correct
                    if not validate_clockwise_order(ordered_vertices):
                        print(f"⚠️  Warning: failed to establish clockwise ordering for polygon in {base_name}")
                        continue

                    # Write label based on format
                    if label_format == 'pose':
                        # Calculate bounding box from keypoints (axis-aligned)
                        x_coords = norm_vertices[:, 0]
                        y_coords = norm_vertices[:, 1]
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()

                        # Bounding box center and size (normalized)
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        # Pose format: 0 x_center y_center width height x1 y1 2 x2 y2 2 x3 y3 2 x4 y4 2
                        keypoints_str = ' '.join([f"{x:.6f} {y:.6f} 2" for x, y in norm_vertices])
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {keypoints_str}\n")
                    else:
                        # Segmentation format: 0 x1 y1 x2 y2 x3 y3 x4 y4
                        coords_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in norm_vertices])
                        f.write(f"0 {coords_str}\n")

                    total_polygons += 1

            total_labels += 1

    print(f"✅ Generated {total_labels} label files ({total_polygons} polygons)")
    return total_labels, total_polygons


def create_train_val_txt() -> Tuple[int, int]:
    """Create train.txt and val.txt with image paths."""
    # Collect training images (3 phones)
    train_images = []
    for phone in TRAIN_PHONES:
        train_images.extend(collect_image_paths(phone))

    # Collect validation images (1 phone)
    val_images = collect_image_paths(VAL_PHONE)

    # Write train.txt
    train_txt = AUGMENTED_DATASET / "train.txt"
    with open(train_txt, 'w') as f:
        for img_path in train_images:
            f.write(f"{img_path}\n")

    # Write val.txt
    val_txt = AUGMENTED_DATASET / "val.txt"
    with open(val_txt, 'w') as f:
        for img_path in val_images:
            f.write(f"{img_path}\n")

    print(f"✅ Created {train_txt}")
    print(f"   - Train images: {len(train_images)} (phones: {', '.join(TRAIN_PHONES)})")
    print(f"✅ Created {val_txt}")
    print(f"   - Val images: {len(val_images)} (phone: {VAL_PHONE})")

    return len(train_images), len(val_images)


def create_yolo_config(config_name: str, description: str) -> Path:
    """Create YOLO dataset configuration file."""
    config = {
        'path': AUGMENTED_DATASET.absolute().as_posix(),
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': 1,
        'names': ['concentration_zone'],
        'kpt_shape': [4, 3]  # 4 keypoints (TL, TR, BR, BL), 3 values each (x, y, visibility)
    }

    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(config_path, 'w') as f:
        f.write(f"# {description}\n")
        f.write(f"# Generated by prepare_yolo_dataset.py\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created {config_path}")
    return config_path


def print_summary(train_count: int, val_count: int, label_count: int, polygon_count: int) -> None:
    """Print dataset summary."""
    print("\n" + "="*60)
    print("YOLO Dataset Preparation Complete")
    print("="*60)
    print(f"Dataset path: {AUGMENTED_DATASET}")
    print(f"Train phones: {', '.join(TRAIN_PHONES)}")
    print(f"Val phone: {VAL_PHONE}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Total images: {train_count + val_count}")
    print(f"YOLO labels: {label_count} files")
    print(f"Total polygons: {polygon_count}")
    print(f"Classes: 1 (concentration_zone)")
    print(f"Config directory: {CONFIGS_DIR}")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate environment: conda activate microPAD-python-env")
    print("2. Train model:")
    print("   python python_scripts/train_yolo.py --stage 1")
    print("   (YOLO scales images to 960x960 at runtime)")
    print("   Or customize resolution:")
    print("   python python_scripts/train_yolo.py --stage 1 --imgsz 960 --batch 24")
    print("="*60)


def verify_labels() -> bool:
    """Verify that label files exist for all images.

    Returns:
        True if all labels exist, False otherwise
    """
    missing_labels = []

    for phone in PHONE_DIRS:
        images = collect_image_paths(phone, use_absolute_paths=True)
        labels_dir = AUGMENTED_DATASET / phone / "labels"

        for img_path in images:
            img_name = Path(img_path).stem
            label_path = labels_dir / f"{img_name}.txt"

            if not label_path.exists():
                missing_labels.append(str(label_path))

    if missing_labels:
        print(f"⚠️  Warning: {len(missing_labels)} label files missing:")
        for label in missing_labels[:5]:
            print(f"   - {label}")
        if len(missing_labels) > 5:
            print(f"   ... and {len(missing_labels) - 5} more")
        return False
    else:
        print(f"✅ All label files verified")
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset for microPAD detection training"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['pose', 'seg'],
        default='pose',
        help="Label format: 'pose' for YOLOv11-pose (default), 'seg' for segmentation (deprecated)"
    )
    return parser.parse_args()


def main() -> None:
    """Main execution.

    Raises:
        FileNotFoundError: If dataset or phone directories not found
        ValueError: If no images found in directories
    """
    args = parse_args()

    print("="*60)
    print("microPAD YOLO Dataset Preparation")
    print("="*60)
    print(f"Label format: {args.format.upper()}")
    if args.format == 'seg':
        print("⚠️  Warning: Segmentation format is deprecated. Pose format recommended.")
    print()

    if not AUGMENTED_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found at {AUGMENTED_DATASET}")

    for phone in PHONE_DIRS:
        phone_path = AUGMENTED_DATASET / phone
        if not phone_path.exists():
            raise FileNotFoundError(f"Phone directory not found: {phone_path}")

    print(f"✅ Dataset found: {AUGMENTED_DATASET}")
    print(f"✅ Phone directories: {', '.join(PHONE_DIRS)}\n")

    print("Restructuring dataset for YOLO...")
    moved_count = restructure_for_yolo()
    if moved_count > 0:
        print(f"✅ Restructured: moved {moved_count} images to images/ subdirectories")
    else:
        print(f"✅ Already restructured: images already in images/ subdirectories")
    print()

    cache_files = list(AUGMENTED_DATASET.glob("*.cache"))
    if cache_files:
        for cache_file in cache_files:
            cache_file.unlink()
        print(f"✅ Deleted {len(cache_files)} cache files")
        print()

    # Generate YOLO labels from MATLAB coordinates
    print("Generating YOLO labels from MATLAB coordinates...")
    label_count, polygon_count = generate_yolo_labels(label_format=args.format)
    print()

    verify_labels()
    print()

    train_count, val_count = create_train_val_txt()
    print()

    create_yolo_config(
        "micropad_synth",
        "microPAD Synthetic Dataset - Train on 3 phones, validate on 1 phone"
    )

    print_summary(train_count, val_count, label_count, polygon_count)


if __name__ == "__main__":
    main()
