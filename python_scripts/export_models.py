#!/usr/bin/env python3
"""Project-specific YOLO export pipeline for ColorflowCapture.

This script exports the two required models to:
- Android TFLite assets (`composeApp/src/androidMain/assets/*.tflite`)
- iOS CoreML packages (`iosApp/iosApp/*.mlpackage`)

It enforces export settings and validates artifact contracts that the app code
expects (input/output shapes, dtypes, and CoreML metadata args).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WORKSPACE_MARKERS = (
    "python_scripts",
    "matlab_scripts",
)

COLORFLOW_PROJECT_MARKERS = (
    "settings.gradle.kts",
    "composeApp/src/commonMain",
    "iosApp/iosApp",
)

DEFAULT_COLORFLOW_APP_DIR = Path.home() / "AndroidStudioProjects" / "ColorflowCapture"
ANDROID_ASSETS_DIR = Path("composeApp/src/androidMain/assets")
IOS_MODELS_DIR = Path("iosApp/iosApp")
DETECTOR_CONFIG_FILE = Path(
    "composeApp/src/commonMain/kotlin/com/colorflow/capture/features/capture/data/config/DetectorConfiguration.kt"
)

# All runtime adapters are hardcoded for 640 input and 8400 anchors.
EXPECTED_INPUT_SIZE = 640
EXPECTED_ANCHORS = 8400


@dataclass(frozen=True)
class ModelSpec:
    key: str
    task: str
    stem: str
    values_per_anchor: int


@dataclass
class ExportedArtifacts:
    spec: ModelSpec
    tflite_path: Path
    coreml_path: Path


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="obb",
        task="obb",
        stem="yolov8n-micropad-obb-640",
        values_per_anchor=6,
    ),
    ModelSpec(
        key="pose",
        task="pose",
        stem="yolov8s-micropad-pose-640",
        values_per_anchor=17,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export and validate ColorflowCapture models with project-locked settings."
        )
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing '<model-stem>.pt' files. "
            "If omitted, script searches common project paths."
        ),
    )
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=DEFAULT_COLORFLOW_APP_DIR,
        help=(
            "ColorflowCapture app repo root (default: "
            "~/AndroidStudioProjects/ColorflowCapture)."
        ),
    )
    parser.add_argument(
        "--obb-weights",
        type=Path,
        default=None,
        help="Explicit path to yolov8n-micropad-obb-640 .pt file.",
    )
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=None,
        help="Explicit path to yolov8s-micropad-pose-640 .pt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Ultralytics export device (default: cpu).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary export workspace for debugging.",
    )
    parser.add_argument(
        "--skip-coreml-metadata-check",
        action="store_true",
        help="Skip strict CoreML metadata args verification.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable extra debug logs.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def info(message: str) -> None:
    print(f"[export_models] {message}", flush=True)


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def resolve_workspace_root(script_path: Path) -> Path:
    script_dir = script_path.resolve().parent
    root = script_dir.parent if script_dir.name == "python_scripts" else script_dir
    missing = [marker for marker in WORKSPACE_MARKERS if not (root / marker).exists()]
    if missing:
        fail(
            "Could not resolve microPAD workspace root from script location. "
            f"Missing markers: {missing}"
        )
    return root


def resolve_colorflow_root(app_dir: Path) -> Path:
    candidate = app_dir.expanduser().resolve()
    if not candidate.exists() or not candidate.is_dir():
        fail(f"ColorflowCapture app directory not found: {candidate}")

    missing = [
        marker for marker in COLORFLOW_PROJECT_MARKERS if not (candidate / marker).exists()
    ]
    if missing:
        fail(
            f"Invalid ColorflowCapture app directory '{candidate}'. "
            f"Missing markers: {missing}"
        )
    return candidate


def configure_runtime_temp_dirs(workspace_root: Path) -> Path:
    candidates: List[Path] = []
    tmpdir_env = os.environ.get("TMPDIR")
    if tmpdir_env:
        candidates.append(Path(tmpdir_env).expanduser())
    candidates.extend([Path(tempfile.gettempdir()), workspace_root / ".tmp"])

    for candidate in candidates:
        if _is_writable_directory(candidate):
            os.environ["TMPDIR"] = str(candidate)
            mpl_cache_dir = candidate / "colorflow_export_models_mpl"
            mpl_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)
            return candidate

    fail("Unable to locate a writable temp directory for export runtime.")


def read_configured_asset_names(app_root: Path) -> List[str]:
    config_path = app_root / DETECTOR_CONFIG_FILE
    if not config_path.exists():
        fail(f"Detector configuration file not found: {config_path}")

    try:
        content = config_path.read_text(encoding="utf-8")
    except Exception as exc:
        fail(f"Failed reading detector configuration '{config_path}': {exc}")

    asset_names = re.findall(r'assetName\s*=\s*"([^"]+)"', content)
    if not asset_names:
        fail(f"No assetName entries found in detector configuration: {config_path}")

    return asset_names


def validate_model_specs_against_app_config(app_root: Path) -> None:
    configured_assets = set(read_configured_asset_names(app_root))
    required_assets = {spec.stem for spec in MODEL_SPECS}

    missing = required_assets - configured_assets
    if missing:
        fail(
            "ColorflowCapture detector configuration is missing required model assets: "
            f"{sorted(missing)}"
        )

    extras = configured_assets - required_assets
    if extras:
        info(
            "WARNING: Detector configuration has additional assets not handled by this script: "
            f"{sorted(extras)}"
        )


def validate_weights_filename(weights_path: Path, spec: ModelSpec) -> None:
    if weights_path.suffix.lower() != ".pt":
        fail(f"{spec.key} weights must be a .pt file: {weights_path}")
    if weights_path.stem != spec.stem:
        fail(
            f"{spec.key} weights filename must be '{spec.stem}.pt' for this export "
            f"pipeline, got: {weights_path.name}"
        )


def check_dependency(module_name: str, import_hint: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - runtime safety path
        fail(
            f"Missing or broken dependency '{module_name}'. "
            f"Install with: {import_hint}. Original error: {exc}"
        )


def resolve_weights_path(
    root: Path,
    spec: ModelSpec,
    explicit_path: Optional[Path],
    weights_dir: Optional[Path],
) -> Path:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.exists():
            fail(f"{spec.key} weights not found: {candidate}")
        validate_weights_filename(candidate, spec)
        return candidate

    search_dirs: List[Path] = []
    if weights_dir is not None:
        search_dirs.append(weights_dir.expanduser().resolve())
    search_dirs.extend(
        [
            root,
            root / "models",
            root / "models" / "weights",
            root / "weights",
        ]
    )

    matches: List[Path] = []
    for directory in search_dirs:
        candidate = directory / f"{spec.stem}.pt"
        if candidate.exists():
            matches.append(candidate.resolve())

    if not matches:
        fail(
            f"Could not find weights for {spec.key} model '{spec.stem}.pt'. "
            f"Use --{spec.key}-weights or --weights-dir."
        )

    unique_matches = sorted(set(matches))
    if len(unique_matches) > 1:
        fail(
            f"Ambiguous weights for {spec.key}: {unique_matches}. "
            f"Pass --{spec.key}-weights explicitly."
        )

    chosen = unique_matches[0]
    validate_weights_filename(chosen, spec)
    return chosen


def parse_metadata_args(raw_args: str) -> Dict[str, Any]:
    if not raw_args:
        return {}

    # Ultralytics stores dict-like text in metadata; support both JSON and repr.
    try:
        value = json.loads(raw_args)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    try:
        value = ast.literal_eval(raw_args)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    return {}


def validate_tflite(
    model_path: Path,
    spec: ModelSpec,
    tf: Any,
) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        fail(f"{model_path.name}: expected 1 input tensor, got {len(input_details)}")
    if len(output_details) != 1:
        fail(f"{model_path.name}: expected 1 output tensor, got {len(output_details)}")

    input_tensor = input_details[0]
    input_shape = tuple(int(v) for v in input_tensor["shape"].tolist())
    input_dtype_name = input_tensor["dtype"].__name__

    if input_shape != (1, EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE, 3):
        fail(
            f"{model_path.name}: invalid input shape {input_shape}, expected "
            f"(1, {EXPECTED_INPUT_SIZE}, {EXPECTED_INPUT_SIZE}, 3)"
        )
    if input_dtype_name != "float32":
        fail(
            f"{model_path.name}: invalid input dtype {input_dtype_name}, expected float32"
        )

    output_tensor = output_details[0]
    output_shape = tuple(int(v) for v in output_tensor["shape"].tolist())
    output_dtype_name = output_tensor["dtype"].__name__

    valid_shapes = {
        (1, spec.values_per_anchor, EXPECTED_ANCHORS),
        (1, EXPECTED_ANCHORS, spec.values_per_anchor),
        (EXPECTED_ANCHORS, spec.values_per_anchor),
    }
    if output_shape not in valid_shapes:
        fail(
            f"{model_path.name}: invalid output shape {output_shape}, expected one of "
            f"{sorted(valid_shapes)}"
        )
    if output_dtype_name != "float32":
        fail(
            f"{model_path.name}: invalid output dtype {output_dtype_name}, expected float32"
        )


def validate_coreml(
    model_path: Path,
    spec: ModelSpec,
    ct: Any,
    strict_metadata: bool,
) -> None:
    model = ct.models.MLModel(str(model_path))
    model_spec = model.get_spec()
    description = model_spec.description

    if len(description.input) != 1:
        fail(
            f"{model_path.name}: expected 1 input feature, got {len(description.input)}"
        )
    if len(description.output) != 1:
        fail(
            f"{model_path.name}: expected 1 output feature, got {len(description.output)}"
        )

    input_feature = description.input[0]
    input_type = input_feature.type.WhichOneof("Type")
    if input_type == "imageType":
        image_type = input_feature.type.imageType
        if image_type.width != EXPECTED_INPUT_SIZE or image_type.height != EXPECTED_INPUT_SIZE:
            fail(
                f"{model_path.name}: invalid image input {image_type.width}x{image_type.height}, "
                f"expected {EXPECTED_INPUT_SIZE}x{EXPECTED_INPUT_SIZE}"
            )
    elif input_type == "multiArrayType":
        in_shape = tuple(int(v) for v in input_feature.type.multiArrayType.shape)
        valid_in_shapes = {
            (1, 3, EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE),
            (1, EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE, 3),
        }
        if in_shape not in valid_in_shapes:
            fail(
                f"{model_path.name}: invalid multiarray input shape {in_shape}, expected one of "
                f"{sorted(valid_in_shapes)}"
            )
    else:
        fail(f"{model_path.name}: unsupported input type '{input_type}'")

    output_feature = description.output[0]
    output_type = output_feature.type.WhichOneof("Type")
    if output_type != "multiArrayType":
        fail(f"{model_path.name}: output must be multiArrayType, got '{output_type}'")

    out_array = output_feature.type.multiArrayType
    out_shape = tuple(int(v) for v in out_array.shape)
    valid_out_shapes = {
        (1, spec.values_per_anchor, EXPECTED_ANCHORS),
        (1, EXPECTED_ANCHORS, spec.values_per_anchor),
        (EXPECTED_ANCHORS, spec.values_per_anchor),
    }
    if out_shape not in valid_out_shapes:
        fail(
            f"{model_path.name}: invalid output shape {out_shape}, expected one of "
            f"{sorted(valid_out_shapes)}"
        )

    float32_dtype = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    float16_dtype = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
    if out_array.dataType not in {float32_dtype, float16_dtype}:
        fail(
            f"{model_path.name}: invalid output dtype {out_array.dataType}, "
            "expected FLOAT32 or FLOAT16"
        )

    if strict_metadata:
        user_defined = description.metadata.userDefined
        args_raw = user_defined.get("args", "")
        args_dict = parse_metadata_args(args_raw)
        if not args_dict:
            fail(
                f"{model_path.name}: missing/invalid CoreML metadata 'args'; got: {args_raw!r}"
            )

        expected_args = {
            "half": True,
            "int8": False,
            "dynamic": False,
            "nms": False,
            "batch": 1,
        }
        for key, expected_value in expected_args.items():
            actual_value = args_dict.get(key)
            if actual_value != expected_value:
                fail(
                    f"{model_path.name}: metadata args['{key}']={actual_value!r}, "
                    f"expected {expected_value!r}"
                )


def locate_tflite_export(export_result: Path, stem: str) -> Path:
    candidates: List[Path] = []

    if export_result.exists() and export_result.is_file() and export_result.suffix == ".tflite":
        candidates.append(export_result)
    if export_result.exists() and export_result.is_dir():
        candidates.extend(export_result.rglob("*.tflite"))

    saved_model_dir = export_result.parent / f"{stem}_saved_model"
    explicit_candidates = [
        saved_model_dir / f"{stem}_float16.tflite",
        saved_model_dir / f"{stem}_float32.tflite",
        saved_model_dir / f"{stem}.tflite",
    ]
    candidates.extend(path for path in explicit_candidates if path.exists())

    if saved_model_dir.exists():
        candidates.extend(saved_model_dir.rglob("*.tflite"))

    if not candidates:
        fail(
            f"Could not locate exported TFLite file for {stem}. "
            f"Exporter returned: {export_result}"
        )

    # Prefer canonical float16 export name, then newest file.
    deduped = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    for preferred in deduped:
        if preferred.name == f"{stem}_float16.tflite":
            return preferred
    return deduped[0]


def locate_coreml_export(export_result: Path, stem: str) -> Path:
    if not export_result.exists():
        fail(f"CoreML export path does not exist: {export_result}")

    if export_result.is_dir() and export_result.suffix == ".mlpackage":
        return export_result

    fail(
        f"CoreML export for {stem} must be a .mlpackage directory, got: {export_result}"
    )


def export_one_model(
    spec: ModelSpec,
    weights_path: Path,
    workspace: Path,
    yolo_cls: Any,
    tf: Any,
    ct: Any,
    device: str,
    strict_metadata: bool,
    verbose: bool,
) -> ExportedArtifacts:
    model_workspace = workspace / spec.stem
    model_workspace.mkdir(parents=True, exist_ok=True)

    staged_weights = model_workspace / f"{spec.stem}.pt"
    shutil.copy2(weights_path, staged_weights)

    info(f"Exporting {spec.key} from {weights_path}")
    model = yolo_cls(str(staged_weights))

    if getattr(model, "task", None) != spec.task:
        fail(
            f"{spec.stem}: loaded task '{getattr(model, 'task', None)}' does not match expected '{spec.task}'"
        )

    export_args = {
        "imgsz": EXPECTED_INPUT_SIZE,
        "half": True,
        "int8": False,
        "batch": 1,
        "dynamic": False,
        "nms": False,
        "device": device,
        "verbose": bool(verbose),
    }

    tflite_result = Path(str(model.export(format="tflite", **export_args))).resolve()
    coreml_result = Path(str(model.export(format="coreml", **export_args))).resolve()

    tflite_path = locate_tflite_export(tflite_result, spec.stem)
    coreml_path = locate_coreml_export(coreml_result, spec.stem)

    validate_tflite(tflite_path, spec, tf)
    validate_coreml(coreml_path, spec, ct, strict_metadata=strict_metadata)

    info(
        f"Validated {spec.key}: "
        f"TFLite={tflite_path.name}, CoreML={coreml_path.name}"
    )

    return ExportedArtifacts(
        spec=spec,
        tflite_path=tflite_path,
        coreml_path=coreml_path,
    )


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def install_with_rollback(
    app_root: Path,
    artifacts: Iterable[ExportedArtifacts],
) -> None:
    android_dir = (app_root / ANDROID_ASSETS_DIR).resolve()
    ios_dir = (app_root / IOS_MODELS_DIR).resolve()

    android_dir.mkdir(parents=True, exist_ok=True)
    ios_dir.mkdir(parents=True, exist_ok=True)

    operations: List[Tuple[Path, Path, bool]] = []
    for item in artifacts:
        operations.append(
            (
                item.tflite_path,
                android_dir / f"{item.spec.stem}.tflite",
                False,
            )
        )
        operations.append(
            (
                item.coreml_path,
                ios_dir / f"{item.spec.stem}.mlpackage",
                True,
            )
        )

    backup_root = Path(tempfile.mkdtemp(prefix="model_backup_", dir=str(app_root)))
    backups: Dict[Path, Path] = {}

    try:
        # Move old artifacts to backup first.
        for _, destination, _ in operations:
            if destination.exists():
                backup_path = backup_root / destination.name
                if backup_path.exists():
                    _remove_path(backup_path)
                shutil.move(str(destination), str(backup_path))
                backups[destination] = backup_path

        # Install new artifacts.
        for source, destination, is_dir in operations:
            if is_dir:
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

    except Exception:
        # Rollback: remove any partially installed artifacts, then restore backups.
        for _, destination, _ in operations:
            if destination.exists():
                _remove_path(destination)

        for destination, backup in backups.items():
            if backup.exists():
                shutil.move(str(backup), str(destination))

        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def verify_installed_artifacts(
    app_root: Path,
    specs: Iterable[ModelSpec],
    tf: Any,
    ct: Any,
    strict_metadata: bool,
) -> None:
    for spec in specs:
        tflite_path = (app_root / ANDROID_ASSETS_DIR / f"{spec.stem}.tflite").resolve()
        coreml_path = (app_root / IOS_MODELS_DIR / f"{spec.stem}.mlpackage").resolve()

        if not tflite_path.exists():
            fail(f"Installed TFLite model missing: {tflite_path}")
        if not coreml_path.exists():
            fail(f"Installed CoreML model missing: {coreml_path}")

        validate_tflite(tflite_path, spec, tf)
        validate_coreml(coreml_path, spec, ct, strict_metadata=strict_metadata)


def main() -> int:
    args = parse_args()
    script_path = Path(__file__)

    try:
        workspace_root = resolve_workspace_root(script_path)
        app_root = resolve_colorflow_root(args.app_dir)
        temp_root = configure_runtime_temp_dirs(workspace_root)

        check_dependency("ultralytics", "pip install ultralytics")
        check_dependency("tensorflow", "pip install tensorflow")
        check_dependency("coremltools", "pip install coremltools")

        from ultralytics import YOLO, __version__ as ultralytics_version
        import tensorflow as tf
        import coremltools as ct

        info(f"Workspace root: {workspace_root}")
        info(f"Colorflow app root: {app_root}")
        info(f"Runtime temp root: {temp_root}")
        info(f"Ultralytics version: {ultralytics_version}")
        validate_model_specs_against_app_config(app_root)

        resolved_weights: Dict[str, Path] = {}
        for spec in MODEL_SPECS:
            explicit = args.obb_weights if spec.key == "obb" else args.pose_weights
            resolved_weights[spec.key] = resolve_weights_path(
                root=workspace_root,
                spec=spec,
                explicit_path=explicit,
                weights_dir=args.weights_dir,
            )
            info(f"Resolved {spec.key} weights: {resolved_weights[spec.key]}")

        if args.keep_temp:
            workspace = Path(tempfile.mkdtemp(prefix="export_models_", dir=str(workspace_root)))
            delete_workspace = False
        else:
            workspace = Path(tempfile.mkdtemp(prefix="export_models_", dir=str(workspace_root)))
            delete_workspace = True

        exported: List[ExportedArtifacts] = []
        for spec in MODEL_SPECS:
            artifact = export_one_model(
                spec=spec,
                weights_path=resolved_weights[spec.key],
                workspace=workspace,
                yolo_cls=YOLO,
                tf=tf,
                ct=ct,
                device=args.device,
                strict_metadata=not args.skip_coreml_metadata_check,
                verbose=args.verbose,
            )
            exported.append(artifact)

        info("Installing exported models into app asset directories...")
        install_with_rollback(app_root=app_root, artifacts=exported)

        info("Verifying installed artifacts...")
        verify_installed_artifacts(
            app_root=app_root,
            specs=MODEL_SPECS,
            tf=tf,
            ct=ct,
            strict_metadata=not args.skip_coreml_metadata_check,
        )

        info("All exports and validations succeeded.")
        info(f"Android assets: {(app_root / ANDROID_ASSETS_DIR).resolve()}")
        info(f"iOS models: {(app_root / IOS_MODELS_DIR).resolve()}")
        return 0

    except Exception as exc:
        info(f"ERROR: {exc}")
        if args.verbose:
            raise
        return 1

    finally:
        workspace = locals().get("workspace")
        delete_workspace = locals().get("delete_workspace")
        if workspace is not None and delete_workspace is not None:
            if delete_workspace:
                shutil.rmtree(workspace, ignore_errors=True)
            else:
                info(f"Kept temp workspace: {workspace}")


if __name__ == "__main__":
    raise SystemExit(main())
