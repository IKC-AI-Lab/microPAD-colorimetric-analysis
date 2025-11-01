#!/usr/bin/env python3
"""
Standalone YOLO inference script for microPAD quad detection.

Called by MATLAB cut_micropads.m to perform AI-based polygon detection.
Accepts image path and outputs detected quad coordinates to stdout.

Usage:
    python detect_quads.py <image_path> <model_path> [--conf THRESHOLD] [--imgsz SIZE]

Output Format (stdout):
    numDetections
    x1 y1 x2 y2 x3 y3 x4 y4 confidence
    x1 y1 x2 y2 x3 y3 x4 y4 confidence
    ...

Note: Coordinates are 0-based (Python/OpenCV convention).
      MATLAB code must add 1 for 1-based indexing.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import cv2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO quad detection for microPAD analysis"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("model_path", type=str, help="Path to YOLO model (.pt)")
    parser.add_argument(
        "--conf", type=float, default=0.6, help="Confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)"
    )
    return parser.parse_args()


def fit_quad_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    """Fit quadrilateral from YOLO contour points using Douglas-Peucker.

    Args:
        contour: Numpy array of contour points (Nx2), float or int dtype

    Returns:
        4x2 numpy array (dtype=float64) of quad vertices (clockwise from top-left),
        or None if fitting failed
    """
    if len(contour) < 4:
        return None

    # Calculate perimeter
    perimeter = cv2.arcLength(contour, closed=True)

    # Try Douglas-Peucker with increasing epsilon (matches MATLAB implementation)
    for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]:
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        if len(approx) == 4:
            # Reshape to (4, 2)
            quad = approx.reshape(4, 2).astype(np.float64)
            return order_corners_clockwise(quad)

    # Fallback: use 4 corners from convex hull or minAreaRect
    hull = cv2.convexHull(contour)
    if len(hull) == 4:
        return order_corners_clockwise(hull.reshape(4, 2).astype(np.float64))

    # Last resort: use minAreaRect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_corners_clockwise(box.astype(np.float64))


def order_corners_clockwise(quad):
    """Order vertices clockwise starting from top-left.

    Top-left is defined as the corner with minimum (x + y).
    Clockwise order from that corner matches MATLAB's orderQuadVertices.

    Args:
        quad: 4x2 numpy array of corner coordinates

    Returns:
        4x2 numpy array ordered clockwise from top-left
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


def detect_quads(image_path, model_path, conf_threshold=0.6, imgsz=640):
    """Run YOLO inference and extract quad coordinates.

    Args:
        image_path: Path to input image
        model_path: Path to YOLO model (.pt file)
        conf_threshold: Confidence threshold for detections
        imgsz: Inference image size

    Returns:
        Tuple of (quads, confidences) where quads is list of 4x2 arrays
    """
    from ultralytics import YOLO

    # Load model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    # Check if any detections exist
    if result.masks is None or len(result.masks.data) == 0:
        return [], []

    quads = []
    confidences = []

    # Process each detection
    for i, (mask_xy, conf) in enumerate(zip(result.masks.xy, result.boxes.conf)):
        # Handle both tensor and numpy array inputs
        if hasattr(mask_xy, 'cpu'):
            contour = mask_xy.cpu().numpy()
        else:
            contour = np.array(mask_xy) if not isinstance(mask_xy, np.ndarray) else mask_xy

        # Fit quadrilateral from contour
        quad = fit_quad_from_contour(contour)

        if quad is not None:
            quads.append(quad)
            # Handle both tensor and scalar confidence values
            if hasattr(conf, 'cpu'):
                confidences.append(float(conf.cpu().item()))
            else:
                confidences.append(float(conf))

    return quads, confidences


def main():
    """Main entry point.

    IMPORTANT: Outputs 0-based pixel coordinates (Python/OpenCV convention).
    MATLAB callers must add 1 to convert to 1-based indexing.
    """
    args = parse_args()

    # Validate inputs
    if not Path(args.image_path).exists():
        print(f"ERROR: Image not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found: {args.model_path}", file=sys.stderr)
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
