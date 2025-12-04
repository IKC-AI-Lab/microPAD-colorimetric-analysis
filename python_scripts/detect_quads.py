#!/usr/bin/env python3
"""
Standalone YOLO inference script for microPAD quad detection using pose keypoints.

Called by MATLAB cut_micropads.m to perform AI-based quad detection.
Accepts image path and outputs detected quad coordinates to stdout.

This script uses YOLOv11s-pose to detect quadrilateral concentration zones on microPAD
images by predicting 4 corner keypoints directly. No polygon simplification is needed.

Model Configuration:
    - Architecture: YOLOv11s-pose (small model for faster inference)
    - Training Resolution: 1280×1280 pixels (optimized for high-res smartphone photos)
    - Default Confidence: 0.6 (60%)

Usage:
    python detect_quads.py <image_path> <model_path> [--conf THRESHOLD] [--imgsz SIZE]

Output Format (stdout):
    numDetections
    x1 y1 x2 y2 x3 y3 x4 y4 confidence
    x1 y1 x2 y2 x3 y3 x4 y4 confidence
    ...

Note: Coordinates are 0-based (Python/OpenCV convention).
      MATLAB code must add 1 for 1-based indexing.
      Keypoints are ordered clockwise from top-left: TL, TR, BR, BL.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np

# Constants
DIVISION_SAFETY_EPSILON = 1e-6  # Minimum denominator to prevent division by zero
DEFAULT_DIMENSION_FALLBACK = 1.0  # Fallback for empty dimension arrays


def parse_args():
    """Parse command line arguments.

    Default values match the trained YOLOv11s-pose model configuration:
    - imgsz=1280: Model was trained at 1280×1280 resolution (optimized for smartphone photos)
    - conf=0.6: Balanced detection threshold
    """
    parser = argparse.ArgumentParser(
        description="YOLOv11s-pose quad detection for microPAD concentration zones"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("model_path", type=str, help="Path to YOLOv11s-pose model (.pt)")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6, recommended range: 0.5-0.7)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size in pixels (default: 1280, matches training resolution)"
    )
    return parser.parse_args()


def order_corners_clockwise(quad: np.ndarray) -> np.ndarray:
    """Order vertices clockwise starting from top-left.

    Top-left is defined as the corner with minimum (x + y).
    Clockwise order from that corner ensures consistent vertex ordering.

    Args:
        quad: 4x2 numpy array of corner coordinates

    Returns:
        4x2 numpy array ordered clockwise from top-left (TL, TR, BR, BL)
    """
    # Find top-left corner (minimum x + y)
    sum_coords = quad[:, 0] + quad[:, 1]  # x + y for each corner
    top_left_idx = np.argmin(sum_coords)

    # Find centroid
    centroid = quad.mean(axis=0)

    # Calculate angles from centroid (for clockwise ordering)
    angles = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])

    # Sort by angle (counter-clockwise from right horizontal)
    order = np.argsort(angles)
    quad_sorted = quad[order]

    # Rotate array to start from top-left
    # Find where top-left ended up in sorted array
    tl_new_pos = np.where(order == top_left_idx)[0][0]
    quad_ordered = np.roll(quad_sorted, -tl_new_pos, axis=0)

    return quad_ordered


def sort_quads_by_layout(quads: List[np.ndarray], confidences: List[float]) -> Tuple[List[np.ndarray], List[float]]:
    """Sort detections from low->high concentration along the dominant strip axis."""
    if len(quads) <= 1:
        return quads, confidences

    quad_array = np.stack(quads).astype(np.float64)

    centroids = quad_array.mean(axis=1)
    min_xy = quad_array.min(axis=1)
    max_xy = quad_array.max(axis=1)

    widths = max_xy[:, 0] - min_xy[:, 0]
    heights = max_xy[:, 1] - min_xy[:, 1]

    width_ref = float(np.median(widths[widths > 0])) if np.any(widths > 0) else DEFAULT_DIMENSION_FALLBACK
    height_ref = float(np.median(heights[heights > 0])) if np.any(heights > 0) else DEFAULT_DIMENSION_FALLBACK

    range_x = float(max_xy[:, 0].max() - min_xy[:, 0].min())
    range_y = float(max_xy[:, 1].max() - min_xy[:, 1].min())

    count_x = range_x / max(width_ref, DIVISION_SAFETY_EPSILON)
    count_y = range_y / max(height_ref, DIVISION_SAFETY_EPSILON)

    if not np.isfinite(count_x):
        count_x = 0.0
    if not np.isfinite(count_y):
        count_y = 0.0

    if count_x >= count_y:
        primary = centroids[:, 0]
        secondary = -centroids[:, 1]
    else:
        primary = -centroids[:, 1]
        secondary = centroids[:, 0]

    order = np.lexsort((secondary, primary))
    quads_sorted = [quads[i] for i in order]
    confidences_sorted = [confidences[i] for i in order]
    return quads_sorted, confidences_sorted


def detect_quads(image_path: str, model_path: str, conf_threshold: float = 0.6, imgsz: int = 1280) -> Tuple[List[np.ndarray], List[float]]:
    """Run YOLOv11s-pose inference and extract concentration zone keypoint coordinates.

    This function runs the trained YOLOv11s-pose model to detect microPAD
    concentration zones. Each detection consists of 4 corner keypoints
    representing a quadrilateral.

    Args:
        image_path: Path to input microPAD image
        model_path: Path to YOLOv11s-pose model (.pt file)
        conf_threshold: Confidence threshold for detections (0.5-0.7 recommended)
        imgsz: Inference image size (default: 1280 to match training resolution)

    Returns:
        Tuple of (quads, confidences) where:
        - quads: List of [4x2] numpy arrays (4 corners, each with x,y coords)
        - confidences: List of detection confidence scores [0, 1]

    Note:
        Detections are returned in YOLO's native order. MATLAB handles spatial sorting
        based on display context (rotation, zoom, memory).
        Optimal performance when imgsz=1280 (training resolution) with ~7 detections per strip.
    """
    from ultralytics import YOLO

    # Load trained pose model
    model = YOLO(model_path)

    # Run inference with optimized settings
    # YOLO automatically detects and uses available GPU
    results = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    # Check if any detections exist
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return [], []

    quads = []
    confidences = []

    # Extract keypoints from pose model
    # result.keypoints.xy shape: [N, 4, 2] for N detections, 4 keypoints, (x, y) coords
    keypoints_xy = result.keypoints.xy.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()

    # Process each detection
    for kpts, conf in zip(keypoints_xy, boxes_conf):
        # kpts shape: [4, 2] for 4 corners
        # Model outputs: TL, TR, BR, BL (clockwise from top-left)

        # Validate all keypoints are visible and finite
        if not np.all(np.isfinite(kpts)):
            continue  # Skip detections with invalid keypoints

        # Ensure consistent clockwise ordering from top-left
        ordered_kpts = order_corners_clockwise(kpts)

        quads.append(ordered_kpts.astype(np.float64))
        confidences.append(float(conf))

    return quads, confidences


def main():
    """Main entry point.

    IMPORTANT: Outputs 0-based pixel coordinates (Python/OpenCV convention).
    MATLAB callers must add 1 to convert to 1-based indexing.

    Expected workflow:
    1. MATLAB calls: python detect_quads.py image.jpg model.pt --conf 0.6 --imgsz 1280
    2. Script loads YOLOv11s-pose model
    3. Runs inference at 1280×1280 resolution (optimized for smartphone photos)
    4. Detects 7 concentration zones (4 corners each)
    5. Outputs coordinates to stdout for MATLAB to parse
    """
    args = parse_args()

    # Validate inputs
    if not Path(args.image_path).exists():
        print(f"ERROR: Image not found: {args.image_path}", file=sys.stderr)
        print(f"  Searched path: {Path(args.image_path).resolve()}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found: {args.model_path}", file=sys.stderr)
        print(f"  Searched path: {Path(args.model_path).resolve()}", file=sys.stderr)
        print(f"  Expected location: models/yolo11s-micropad-pose-1280.pt", file=sys.stderr)
        sys.exit(1)

    # Validate confidence threshold
    if not 0.0 <= args.conf <= 1.0:
        print(f"ERROR: Invalid confidence threshold: {args.conf}", file=sys.stderr)
        print(f"  Must be in range [0.0, 1.0]. Recommended: 0.5-0.7", file=sys.stderr)
        sys.exit(1)

    try:
        # Run detection
        quads, confidences = detect_quads(
            args.image_path,
            args.model_path,
            args.conf,
            args.imgsz
        )

        # Output format: number of detections followed by quad data
        print(len(quads))

        for quad, conf in zip(quads, confidences):
            # Flatten quad to single line: x1 y1 x2 y2 x3 y3 x4 y4 confidence
            coords = ' '.join([f"{x:.6f} {y:.6f}" for x, y in quad])
            print(f"{coords} {conf:.6f}")

    except (RuntimeError, ValueError, OSError, IOError) as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
