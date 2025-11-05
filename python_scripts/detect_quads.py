#!/usr/bin/env python3
"""
Standalone YOLO inference script for microPAD quad detection using pose keypoints.

Called by MATLAB cut_micropads.m to perform AI-based polygon detection.
Accepts image path and outputs detected quad coordinates to stdout.

This script uses YOLOv11m-pose to detect quadrilateral concentration zones on microPAD
images by predicting 4 corner keypoints directly. No polygon simplification is needed.

Model Configuration:
    - Architecture: YOLOv11m-pose
    - Training Resolution: 960×960 pixels
    - Training Performance: 94.19% mAP50-95, 96.47% precision, 92.27% recall
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


def parse_args():
    """Parse command line arguments.

    Default values match the trained YOLOv11m-pose model configuration:
    - imgsz=960: Model was trained at 960×960 resolution
    - conf=0.6: Balanced threshold for 94.2% mAP model (96.5% precision, 92.3% recall)
    """
    parser = argparse.ArgumentParser(
        description="YOLOv11m-pose quad detection for microPAD concentration zones"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("model_path", type=str, help="Path to YOLOv11m-pose model (.pt)")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6, recommended range: 0.5-0.7)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size in pixels (default: 960, matches training resolution)"
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


def detect_quads(image_path: str, model_path: str, conf_threshold: float = 0.6, imgsz: int = 960) -> Tuple[List[np.ndarray], List[float]]:
    """Run YOLOv11m-pose inference and extract concentration zone keypoint coordinates.

    This function runs the trained YOLOv11m-pose model (94.19% mAP50-95) to detect
    microPAD concentration zones. Each detection consists of 4 corner keypoints
    representing a quadrilateral polygon.

    Args:
        image_path: Path to input microPAD image
        model_path: Path to YOLOv11m-pose model (.pt file)
        conf_threshold: Confidence threshold for detections (0.5-0.7 recommended)
        imgsz: Inference image size (default: 960 to match training resolution)

    Returns:
        Tuple of (quads, confidences) where:
        - quads: List of [4x2] numpy arrays (4 corners, each with x,y coords)
        - confidences: List of detection confidence scores [0, 1]

    Note:
        Optimal performance when imgsz=960 (training resolution).
        Expected output: 7 detections per microPAD strip at conf=0.6.
    """
    from ultralytics import YOLO

    # Load trained pose model
    model = YOLO(model_path)

    # Run inference with optimized settings
    results = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False,
        device='cuda' if model.device.type == 'cuda' else 'cpu'  # Auto-detect GPU
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
    1. MATLAB calls: python detect_quads.py image.jpg model.pt --conf 0.6 --imgsz 960
    2. Script loads YOLOv11m-pose model (94.2% mAP)
    3. Runs inference at 960×960 resolution
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
        print(f"  Expected location: models/yolo11m_micropad_pose.pt", file=sys.stderr)
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

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
