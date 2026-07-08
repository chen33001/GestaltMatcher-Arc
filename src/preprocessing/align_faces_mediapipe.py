"""
Align frontal and profile faces with MediaPipe.

Default CLI:
    python src/preprocessing/align_faces_mediapipe.py ^
      --input data/GestaltMatcherDB/v1.1.3/gmdb ^
      --output data/GestaltMatcherDB/v1.1.3/gmdb_align_mediapipe ^
      --view profile ^
      --metadata_dir data/GestaltMatcherDB/v1.1.3/metadata

Behavior:
    - If a canonical view-aware metadata file exists under metadata_dir, it is
      used as the input metadata and filtered by --view.
    - Otherwise, the script scans --input directly and creates minimal metadata
      rows using the provided --view value.
    - Output metadata and skipped CSV are written into --metadata_dir.
"""

import argparse
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import transform as trans


ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

RIGHT_EYE_INDICES = [33, 133, 159, 145]
LEFT_EYE_INDICES = [362, 263, 386, 374]
NOSE_TIP_INDEX = 1
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROFILE_ROI_EXPAND_X = 0.65
PROFILE_ROI_EXPAND_Y = 0.40
PROFILE_CANONICAL_DIRECTION = "left"
PROFILE_FRONT_MARGIN_RATIO = 0.35
PROFILE_EAR_MARGIN_RATIO = 0.65
PROFILE_TOP_MARGIN_RATIO = 0.45
PROFILE_BOTTOM_MARGIN_RATIO = 0.35
PROFILE_DIRECTION_UNKNOWN_MARGIN = 0.08
PROFILE_HEURISTIC_HIGH_THRESHOLD = 0.18
PROFILE_HEURISTIC_MEDIUM_THRESHOLD = 0.12
PROFILE_DETECTOR_ROI_DIRECTION_EXPAND_X = 0.25
PROFILE_DETECTOR_ROI_DIRECTION_EXPAND_Y = 0.20


def parse_args():
    parser = argparse.ArgumentParser(description="Crop frontal/profile faces with MediaPipe.")

    parser.add_argument("--input", required=True, help="Raw image file or directory.")
    parser.add_argument("--output", required=True, help="Aligned image output directory.")
    parser.add_argument("--view", required=True, choices=["frontal", "profile"], help="Face view to process.")
    parser.add_argument(
        "--metadata_dir",
        required=True,
        help="Directory containing source metadata and receiving output metadata/skipped CSVs.",
    )

    parser.add_argument(
        "--metadata_csv",
        default="",
        help="Optional explicit input metadata CSV. Defaults to <metadata_dir>/view_aware_metadata.csv if present.",
    )
    parser.add_argument(
        "--output_metadata_name",
        default="",
        help="Optional output metadata filename. Defaults to view_aware_metadata_mediapipe_<view>.csv.",
    )
    parser.add_argument(
        "--skipped_csv_name",
        default="",
        help="Optional skipped CSV filename. Defaults to view_aware_metadata_mediapipe_<view>_skipped.csv.",
    )
    parser.add_argument("--image_id_col", default="image_id", help="Metadata image ID column.")
    parser.add_argument("--image_path_col", default="source_path", help="Metadata raw image path column.")
    parser.add_argument("--output_size", type=int, default=112, help="Aligned crop size.")
    parser.add_argument("--alignment_method", default="mediapipe", help="Metadata alignment method value.")
    parser.add_argument(
        "--face_landmarker_model",
        default="saved_models/mediapipe_models/face_landmarker.task",
        help="Path to face_landmarker.task for MediaPipe Tasks API.",
    )
    parser.add_argument(
        "--face_detector_model",
        default="saved_models/mediapipe_models/blaze_face_short_range.tflite",
        help="Optional fallback face detector model path.",
    )
    parser.add_argument(
        "--profile_crop_mode",
        choices=["expanded_bbox", "expanded_bbox_no_pad"],
        default="expanded_bbox_no_pad",
        help="Crop strategy used for profile rows.",
    )
    parser.add_argument("--profile_padding_x", type=float, default=0.8, help="Horizontal profile padding.")
    parser.add_argument("--profile_padding_y", type=float, default=0.45, help="Vertical profile padding.")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing aligned images.")
    parser.add_argument("--max_images", type=int, default=0, help="Optional limit for debugging.")
    return parser.parse_args()


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def collect_image_paths(input_path):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        candidates = []
        for candidate in glob(str(path / "**" / "*"), recursive=True):
            candidate_path = Path(candidate)
            if candidate_path.is_file() and candidate_path.suffix.lower() in IMAGE_EXTENSIONS:
                candidates.append(candidate_path)
        return sorted(set(candidates))
    return []


def metadata_from_data_paths(input_path, view):
    rows = []
    for image_path in collect_image_paths(input_path):
        rows.append(
            {
                "image_id": image_path.stem,
                "view": view,
                "source_path": str(image_path),
                "flat_filename": image_path.name,
            }
        )
    return pd.DataFrame(rows)


def input_metadata_path(args, metadata_dir):
    if args.metadata_csv:
        return Path(args.metadata_csv)
    candidates = [
        metadata_dir / "view_aware_metadata.csv",
        metadata_dir / "core" / "view_aware_metadata.csv",
        metadata_dir / "legacy" / "view_aware_metadata.csv",
        metadata_dir / "legacy" / "gmdb_view_aware_metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def output_metadata_path(args, metadata_dir):
    filename = args.output_metadata_name or f"view_aware_metadata_mediapipe_{args.view}.csv"
    return metadata_dir / filename


def skipped_metadata_path(args, metadata_dir):
    filename = args.skipped_csv_name or f"view_aware_metadata_mediapipe_{args.view}_skipped.csv"
    return metadata_dir / filename


def infer_dataset_root(input_path, metadata_dir):
    input_candidate = Path(input_path)
    if input_candidate.is_dir():
        return input_candidate.parent if input_candidate.name.lower() in {"gmdb", "gmdb_frontal", "gmdb_profile"} else input_candidate
    return metadata_dir.parent


def resolve_image_path(row, args, dataset_root):
    candidates = []
    if args.image_path_col in row.index and not pd.isna(row[args.image_path_col]):
        raw_value = str(row[args.image_path_col]).strip()
        if raw_value:
            raw_path = Path(raw_value)
            candidates.append(raw_path)
            if not raw_path.is_absolute():
                candidates.append(Path.cwd() / raw_path)
                candidates.append(dataset_root / raw_path)

    image_id = str(row[args.image_id_col]).strip()
    view = str(row.get("view", args.view)).strip().lower()
    if view == "frontal":
        candidates.append(dataset_root / "gmdb_frontal" / f"{image_id}.jpg")
    elif view == "profile":
        candidates.append(dataset_root / "gmdb_profile" / f"{image_id}.jpg")
    candidates.extend(
        [
            dataset_root / "gmdb" / f"{image_id}.jpg",
            dataset_root / "gmdb_frontal" / f"{image_id}.jpg",
            dataset_root / "gmdb_profile" / f"{image_id}.jpg",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def landmarks_to_points(face_landmarks, width, height):
    landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks

    def point(index):
        landmark = landmarks[index]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    def mean_point(indices):
        return np.mean([point(index) for index in indices], axis=0)

    return np.array(
        [
            mean_point(RIGHT_EYE_INDICES),
            mean_point(LEFT_EYE_INDICES),
            point(NOSE_TIP_INDEX),
            point(MOUTH_LEFT_INDEX),
            point(MOUTH_RIGHT_INDEX),
        ],
        dtype=np.float32,
    )


def all_landmarks_to_points(face_landmarks, width, height):
    landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    return np.array([[landmark.x * width, landmark.y * height] for landmark in landmarks], dtype=np.float32)


def estimate_norm(landmarks, image_size):
    src = ARCFACE_SRC if image_size == 112 else float(image_size) / 112 * ARCFACE_SRC
    tform = trans.SimilarityTransform()
    if not tform.estimate(landmarks, src):
        return None
    return tform.params[0:2, :]


def align_image(image_bgr, landmarks, output_size):
    matrix = estimate_norm(landmarks, output_size)
    if matrix is None:
        return None
    return cv2.warpAffine(image_bgr, matrix, (output_size, output_size), borderValue=0.0)


def crop_expanded_bbox(image_bgr, all_landmarks, output_size, padding_x, padding_y):
    height, width = image_bgr.shape[:2]
    x_min, y_min = np.min(all_landmarks, axis=0)
    x_max, y_max = np.max(all_landmarks, axis=0)

    box_w = max(float(x_max - x_min), 1.0)
    box_h = max(float(y_max - y_min), 1.0)
    center_x = float((x_min + x_max) / 2.0)
    center_y = float((y_min + y_max) / 2.0)

    crop_w = box_w * (1.0 + padding_x)
    crop_h = box_h * (1.0 + padding_y)
    side = int(np.ceil(max(crop_w, crop_h)))

    x1 = int(np.floor(center_x - side / 2.0))
    y1 = int(np.floor(center_y - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(width, x2)
    y2_clamped = min(height, y2)

    crop = image_bgr[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    if crop.size == 0:
        return None

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )

    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def crop_expanded_bbox_no_pad(image_bgr, all_landmarks, output_size, padding_x, padding_y):
    height, width = image_bgr.shape[:2]
    x_min, y_min = np.min(all_landmarks, axis=0)
    x_max, y_max = np.max(all_landmarks, axis=0)

    box_w = max(float(x_max - x_min), 1.0)
    box_h = max(float(y_max - y_min), 1.0)
    center_x = float((x_min + x_max) / 2.0)
    center_y = float((y_min + y_max) / 2.0)

    crop_w = min(box_w * (1.0 + padding_x), float(width))
    crop_h = min(box_h * (1.0 + padding_y), float(height))

    x1 = int(np.floor(center_x - crop_w / 2.0))
    y1 = int(np.floor(center_y - crop_h / 2.0))
    x2 = int(np.ceil(center_x + crop_w / 2.0))
    y2 = int(np.ceil(center_y + crop_h / 2.0))

    if x1 < 0:
        x2 = min(width, x2 - x1)
        x1 = 0
    if y1 < 0:
        y2 = min(height, y2 - y1)
        y1 = 0
    if x2 > width:
        x1 = max(0, x1 - (x2 - width))
        x2 = width
    if y2 > height:
        y1 = max(0, y1 - (y2 - height))
        y2 = height

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def rotate_image_and_points(image_bgr, points, angle_degrees, center):
    matrix = cv2.getRotationMatrix2D(tuple(center), angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(
        image_bgr,
        matrix,
        image_bgr.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )
    rotated_points = cv2.transform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)
    return rotated_image, rotated_points


def estimate_profile_roll(all_landmarks):
    centered = all_landmarks - np.mean(all_landmarks, axis=0, keepdims=True)
    if len(centered) < 2:
        return 0.0

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    principal_axis = vt[0]
    angle = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))

    # Normalize to the closest vertical axis so we only correct in-plane roll.
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    if angle >= 0.0:
        target = 90.0
    else:
        target = -90.0
    return target - angle


def autorotate_profile_image(image_bgr, all_landmarks):
    center = np.mean(all_landmarks, axis=0)
    rotation = estimate_profile_roll(all_landmarks)
    if abs(rotation) < 1.0:
        return image_bgr, all_landmarks, 0.0
    rotated_image, rotated_landmarks = rotate_image_and_points(image_bgr, all_landmarks, rotation, center)
    return rotated_image, rotated_landmarks, rotation


def clamp_box_to_image(x1, y1, x2, y2, image_shape):
    height, width = image_shape[:2]
    return (
        max(0, min(int(np.floor(x1)), width)),
        max(0, min(int(np.floor(y1)), height)),
        max(0, min(int(np.ceil(x2)), width)),
        max(0, min(int(np.ceil(y2)), height)),
    )


def expand_detector_bbox_for_profile_roi(bbox, image_shape, expand_x=PROFILE_ROI_EXPAND_X, expand_y=PROFILE_ROI_EXPAND_Y):
    x, y, w, h = bbox
    x1 = x - w * expand_x
    x2 = x + w * (1.0 + expand_x)
    y1 = y - h * expand_y
    y2 = y + h * (1.0 + expand_y)
    return clamp_box_to_image(x1, y1, x2, y2, image_shape)


def extract_roi(image_bgr, roi_box):
    x1, y1, x2, y2 = roi_box
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    return roi, np.array([x1, y1], dtype=np.float32)


def remap_landmarks_from_roi(roi_landmarks, roi_origin_xy):
    return roi_landmarks + roi_origin_xy.reshape(1, 2)


def retry_profile_landmarks_on_detector_roi(image_bgr, detector, bbox):
    roi_box = expand_detector_bbox_for_profile_roi(bbox, image_bgr.shape)
    roi_bgr, roi_origin = extract_roi(image_bgr, roi_box)
    if roi_bgr is None:
        return None
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_face_landmarks = detector.detect(roi_rgb)
    if roi_face_landmarks is None:
        return None
    roi_height, roi_width = roi_bgr.shape[:2]
    roi_points = all_landmarks_to_points(roi_face_landmarks, roi_width, roi_height)
    return remap_landmarks_from_roi(roi_points, roi_origin)


def detect_profile_geometry(image_bgr, image_rgb, detector, bbox_detector, initial_face_landmarks, args):
    width, height = image_bgr.shape[1], image_bgr.shape[0]
    debug = {
        "profile_full_landmarks_found": initial_face_landmarks is not None,
        "profile_detector_bbox_found": False,
        "profile_roi_retry_attempted": False,
        "profile_roi_retry_succeeded": False,
    }
    if initial_face_landmarks is not None:
        return {
            "geometry_source": "landmarks",
            "landmarks": all_landmarks_to_points(initial_face_landmarks, width, height),
            "bbox": None,
            "debug": debug,
        }

    if bbox_detector is None:
        return {"geometry_source": "none", "landmarks": None, "bbox": None, "debug": debug}

    bbox = bbox_detector.detect_bbox(image_rgb)
    if bbox is None:
        return {"geometry_source": "none", "landmarks": None, "bbox": None, "debug": debug}

    debug["profile_detector_bbox_found"] = True
    debug["detector_bbox"] = bbox
    debug["profile_roi_retry_attempted"] = True
    retry_landmarks = retry_profile_landmarks_on_detector_roi(image_bgr, detector, bbox)
    if retry_landmarks is not None:
        debug["profile_roi_retry_succeeded"] = True
        return {
            "geometry_source": "landmarks_roi_retry",
            "landmarks": retry_landmarks,
            "bbox": bbox,
            "debug": debug,
        }

    return {"geometry_source": "detector_bbox", "landmarks": None, "bbox": bbox, "debug": debug}


def estimate_profile_direction_from_landmarks(all_landmarks):
    if all_landmarks is None or len(all_landmarks) <= NOSE_TIP_INDEX:
        return {"direction": "unknown", "confidence": "unknown", "source": "landmarks"}
    nose_x = float(all_landmarks[NOSE_TIP_INDEX][0])
    x_min = float(np.min(all_landmarks[:, 0]))
    x_max = float(np.max(all_landmarks[:, 0]))
    spread_x = max(x_max - x_min, 1.0)
    normalized = (nose_x - x_min) / spread_x
    offset = normalized - 0.5
    if abs(offset) < PROFILE_DIRECTION_UNKNOWN_MARGIN:
        return {"direction": "unknown", "confidence": "low", "source": "landmarks"}
    confidence = "high" if abs(offset) >= 0.18 else "medium"
    direction = "left" if offset < 0.0 else "right"
    return {"direction": direction, "confidence": confidence, "source": "landmarks"}


def flip_image_and_points_horizontally(image_bgr, points):
    flipped = cv2.flip(image_bgr, 1)
    width = image_bgr.shape[1]
    flipped_points = points.copy()
    flipped_points[:, 0] = (width - 1) - flipped_points[:, 0]
    return flipped, flipped_points


def flip_image_and_bbox_horizontally(image_bgr, bbox):
    flipped = cv2.flip(image_bgr, 1)
    width = image_bgr.shape[1]
    x, y, w, h = bbox
    flipped_x = width - (x + w)
    return flipped, (flipped_x, y, w, h)


def extract_profile_detector_roi(
    image_bgr,
    bbox,
    expand_x=PROFILE_DETECTOR_ROI_DIRECTION_EXPAND_X,
    expand_y=PROFILE_DETECTOR_ROI_DIRECTION_EXPAND_Y,
):
    roi_box = expand_detector_bbox_for_profile_roi(bbox, image_bgr.shape, expand_x=expand_x, expand_y=expand_y)
    roi_bgr, _ = extract_roi(image_bgr, roi_box)
    return {"roi_bgr": roi_bgr, "roi_box": roi_box}


def estimate_detector_direction_center_mass(roi_bgr):
    if roi_bgr is None:
        return {
            "direction": "unknown",
            "confidence": "unknown",
            "source": "detector_center_mass",
            "score": np.nan,
        }

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    if edges.sum() == 0:
        return {
            "direction": "unknown",
            "confidence": "unknown",
            "source": "detector_center_mass",
            "score": np.nan,
        }

    edge_weights = edges.astype(np.float32)
    xs = np.arange(edge_weights.shape[1], dtype=np.float32)
    total_weight = float(edge_weights.sum())
    center_x = float((edge_weights.sum(axis=0) * xs).sum() / total_weight)
    score = (center_x / max(edge_weights.shape[1] - 1, 1)) - 0.5

    if abs(score) < PROFILE_HEURISTIC_MEDIUM_THRESHOLD:
        return {"direction": "unknown", "confidence": "low", "source": "detector_center_mass", "score": score}
    if abs(score) >= PROFILE_HEURISTIC_HIGH_THRESHOLD:
        confidence = "high"
    else:
        confidence = "medium"

    # Calibrated on the local GMDB profile set: this score sign maps opposite to the initial assumption.
    direction = "left" if score < 0.0 else "right"
    return {"direction": direction, "confidence": confidence, "source": "detector_center_mass", "score": score}


def estimate_detector_direction_edge_balance(roi_bgr):
    if roi_bgr is None:
        return {
            "direction": "unknown",
            "confidence": "unknown",
            "source": "detector_edge_balance",
            "score": np.nan,
        }

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160).astype(np.float32)
    if edges.sum() == 0:
        return {
            "direction": "unknown",
            "confidence": "unknown",
            "source": "detector_edge_balance",
            "score": np.nan,
        }

    mid = max(1, edges.shape[1] // 2)
    left_edges = edges[:, :mid]
    right_edges = edges[:, mid:]
    left_density = float(left_edges.mean()) if left_edges.size else 0.0
    right_density = float(right_edges.mean()) if right_edges.size else 0.0
    denom = left_density + right_density + 1e-6
    score = (left_density - right_density) / denom

    if abs(score) < PROFILE_HEURISTIC_MEDIUM_THRESHOLD:
        return {"direction": "unknown", "confidence": "low", "source": "detector_edge_balance", "score": score}
    if abs(score) >= PROFILE_HEURISTIC_HIGH_THRESHOLD:
        confidence = "high"
    else:
        confidence = "medium"

    # Calibrated on the local GMDB profile set: this score sign maps opposite to the initial assumption.
    direction = "left" if score > 0.0 else "right"
    return {"direction": direction, "confidence": confidence, "source": "detector_edge_balance", "score": score}


def merge_detector_direction_votes(center_mass_vote, edge_balance_vote):
    def confidence_rank(value):
        return {"unknown": 0, "low": 1, "medium": 2, "high": 3}.get(value, 0)

    directions = [center_mass_vote["direction"], edge_balance_vote["direction"]]
    non_unknown = [direction for direction in directions if direction != "unknown"]
    if not non_unknown:
        final_direction = "unknown"
        final_confidence = "unknown"
    elif len(set(non_unknown)) > 1:
        final_direction = "unknown"
        final_confidence = "low"
    elif len(non_unknown) == 2:
        final_direction = non_unknown[0]
        final_confidence = (
            "high"
            if max(confidence_rank(center_mass_vote["confidence"]), confidence_rank(edge_balance_vote["confidence"])) >= 3
            else "medium"
        )
    else:
        final_direction = "unknown"
        final_confidence = "low"

    return {
        "direction": final_direction,
        "confidence": final_confidence,
        "source": "detector_heuristic",
        "votes": {
            "center_mass": center_mass_vote,
            "edge_balance": edge_balance_vote,
        },
    }


def estimate_profile_direction_from_detector_roi(image_bgr, bbox):
    roi = extract_profile_detector_roi(image_bgr, bbox)
    center_mass_vote = estimate_detector_direction_center_mass(roi["roi_bgr"])
    edge_balance_vote = estimate_detector_direction_edge_balance(roi["roi_bgr"])
    merged = merge_detector_direction_votes(center_mass_vote, edge_balance_vote)
    merged["debug"] = {
        "profile_detector_center_mass_score": center_mass_vote["score"],
        "profile_detector_center_mass_direction": center_mass_vote["direction"],
        "profile_detector_center_mass_confidence": center_mass_vote["confidence"],
        "profile_detector_edge_balance_score": edge_balance_vote["score"],
        "profile_detector_edge_balance_direction": edge_balance_vote["direction"],
        "profile_detector_edge_balance_confidence": edge_balance_vote["confidence"],
    }
    return merged


def resolve_profile_direction(all_landmarks, image_bgr, bbox):
    if all_landmarks is not None:
        result = estimate_profile_direction_from_landmarks(all_landmarks)
        result["debug"] = {}
        return result
    if bbox is not None:
        return estimate_profile_direction_from_detector_roi(image_bgr, bbox)
    return {"direction": "unknown", "confidence": "unknown", "source": "unknown", "debug": {}}


def normalize_profile_pose(image_bgr, all_landmarks=None, bbox=None, canonical_direction=PROFILE_CANONICAL_DIRECTION):
    result = {
        "image_bgr": image_bgr,
        "landmarks": all_landmarks,
        "bbox": bbox,
        "rotation_degrees": 0.0,
        "flipped": False,
        "direction_inferred": "unknown",
        "direction_source": "unknown",
        "direction_confidence": "unknown",
        "direction_debug": {},
    }

    if all_landmarks is not None:
        rotated_image, rotated_landmarks, rotation = autorotate_profile_image(image_bgr, all_landmarks)
        result["image_bgr"] = rotated_image
        result["landmarks"] = rotated_landmarks
        result["rotation_degrees"] = rotation
        direction = resolve_profile_direction(rotated_landmarks, rotated_image, None)
        result["direction_inferred"] = direction["direction"]
        result["direction_source"] = direction["source"]
        result["direction_confidence"] = direction["confidence"]
        result["direction_debug"] = direction.get("debug", {})
        # Flip is intentionally disabled for now.
        # The logic below is kept as a handoff point for future work once
        # detector/landmark direction inference is validated on a larger sample.
        #
        # if direction["direction"] != "unknown" and direction["direction"] != canonical_direction:
        #     flipped_image, flipped_landmarks = flip_image_and_points_horizontally(rotated_image, rotated_landmarks)
        #     result["image_bgr"] = flipped_image
        #     result["landmarks"] = flipped_landmarks
        #     result["flipped"] = True
        return result

    if bbox is not None:
        direction = resolve_profile_direction(None, image_bgr, bbox)
        result["direction_inferred"] = direction["direction"]
        result["direction_source"] = direction["source"]
        result["direction_confidence"] = direction["confidence"]
        result["direction_debug"] = direction.get("debug", {})
        # Flip is intentionally disabled for now.
        # The logic below is kept as a handoff point for future work once
        # detector-side direction inference is calibrated.
        #
        # if direction["direction"] != "unknown" and direction["direction"] != canonical_direction:
        #     flipped_image, flipped_bbox = flip_image_and_bbox_horizontally(image_bgr, bbox)
        #     result["image_bgr"] = flipped_image
        #     result["bbox"] = flipped_bbox
        #     result["flipped"] = True
    return result


def square_box_from_rect(x1, y1, x2, y2):
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    side = max(width, height)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return center_x - side / 2.0, center_y - side / 2.0, center_x + side / 2.0, center_y + side / 2.0


def build_profile_crop_box_from_landmarks(all_landmarks, image_shape, args):
    x_min, y_min = np.min(all_landmarks, axis=0)
    x_max, y_max = np.max(all_landmarks, axis=0)
    face_w = max(float(x_max - x_min), 1.0)
    face_h = max(float(y_max - y_min), 1.0)

    x1 = x_min - face_w * PROFILE_FRONT_MARGIN_RATIO
    x2 = x_max + face_w * PROFILE_EAR_MARGIN_RATIO
    y1 = y_min - face_h * PROFILE_TOP_MARGIN_RATIO
    y2 = y_max + face_h * PROFILE_BOTTOM_MARGIN_RATIO

    return clamp_box_to_image(*square_box_from_rect(x1, y1, x2, y2), image_shape)


def build_profile_crop_box_from_bbox(bbox, image_shape, args):
    x, y, w, h = bbox
    x1 = x - w * PROFILE_FRONT_MARGIN_RATIO
    x2 = x + w * (1.0 + PROFILE_EAR_MARGIN_RATIO)
    y1 = y - h * PROFILE_TOP_MARGIN_RATIO
    y2 = y + h * (1.0 + PROFILE_BOTTOM_MARGIN_RATIO)
    return clamp_box_to_image(*square_box_from_rect(x1, y1, x2, y2), image_shape)


def render_crop_box(image_bgr, crop_box, output_size):
    x1, y1, x2, y2 = crop_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image_bgr.shape[1])
    pad_bottom = max(0, y2 - image_bgr.shape[0])

    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(image_bgr.shape[1], x2)
    y2_clamped = min(image_bgr.shape[0], y2)
    crop = image_bgr[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    if crop.size == 0:
        return None

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )

    if crop.shape[0] != height or crop.shape[1] != width:
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def crop_profile_image_v2(image_bgr, image_rgb, detector, bbox_detector, initial_face_landmarks, args, out_row):
    geometry = detect_profile_geometry(image_bgr, image_rgb, detector, bbox_detector, initial_face_landmarks, args)
    out_row["profile_geometry_source"] = geometry["geometry_source"]
    for key, value in geometry["debug"].items():
        if key != "detector_bbox":
            out_row[key] = value

    if geometry["geometry_source"] == "none":
        out_row["crop_mode"] = "profile_bbox_detector_failed" if bbox_detector is not None else "profile_landmarks_not_found"
        return None

    pose = normalize_profile_pose(
        image_bgr,
        all_landmarks=geometry["landmarks"],
        bbox=geometry["bbox"],
        canonical_direction=PROFILE_CANONICAL_DIRECTION,
    )
    out_row["profile_rotation_degrees"] = pose["rotation_degrees"]
    out_row["profile_flipped"] = pose["flipped"]
    out_row["profile_direction_inferred"] = pose["direction_inferred"]
    out_row["profile_direction_source"] = pose["direction_source"]
    out_row["profile_direction_confidence"] = pose["direction_confidence"]
    for key, value in pose.get("direction_debug", {}).items():
        out_row[key] = value

    if geometry["geometry_source"] in {"landmarks", "landmarks_roi_retry"}:
        crop_box = build_profile_crop_box_from_landmarks(pose["landmarks"], pose["image_bgr"].shape, args)
        out_row["crop_mode"] = (
            "profile_landmarks_box_v2"
            if geometry["geometry_source"] == "landmarks"
            else "profile_landmarks_roi_retry_box_v2"
        )
    else:
        crop_box = build_profile_crop_box_from_bbox(pose["bbox"], pose["image_bgr"].shape, args)
        out_row["crop_mode"] = "profile_detector_box_v2"

    return render_crop_box(pose["image_bgr"], crop_box, args.output_size)


def relative_or_absolute(path):
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


class LegacyFaceMeshDetector:
    def __init__(self, face_mesh_module, args):
        self.face_mesh = face_mesh_module.FaceMesh(
            static_image_mode=True,
            max_num_faces=args.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=args.min_detection_confidence,
        )

    def detect(self, image_rgb):
        result = self.face_mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0]

    def close(self):
        self.face_mesh.close()


class TasksFaceLandmarkerDetector:
    def __init__(self, mp_module, args):
        model_path = Path(args.face_landmarker_model)
        if not model_path.exists():
            raise SystemExit(f"Face Landmarker model not found: {model_path}")

        BaseOptions = mp_module.tasks.BaseOptions
        FaceLandmarker = mp_module.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp_module.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp_module.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=args.max_num_faces,
            min_face_detection_confidence=args.min_detection_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.mp = mp_module

    def detect(self, image_rgb):
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        self.landmarker.close()


class TasksFaceDetector:
    def __init__(self, mp_module, args):
        model_path = Path(args.face_detector_model)
        if not model_path.exists():
            raise SystemExit(f"Face Detector model not found: {model_path}")

        BaseOptions = mp_module.tasks.BaseOptions
        FaceDetector = mp_module.tasks.vision.FaceDetector
        FaceDetectorOptions = mp_module.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp_module.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=args.min_detection_confidence,
        )
        self.detector = FaceDetector.create_from_options(options)
        self.mp = mp_module

    def detect_bbox(self, image_rgb):
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if not result.detections:
            return None
        detection = max(result.detections, key=lambda item: item.bounding_box.width * item.bounding_box.height)
        bbox = detection.bounding_box
        return bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    def close(self):
        self.detector.close()


def load_face_detector(args):
    try:
        import mediapipe.solutions.face_mesh as face_mesh

        return LegacyFaceMeshDetector(face_mesh, args)
    except ImportError:
        pass

    try:
        from mediapipe.python.solutions import face_mesh

        return LegacyFaceMeshDetector(face_mesh, args)
    except ImportError:
        pass

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("MediaPipe is not installed. Install it before running this script.") from exc

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return LegacyFaceMeshDetector(mp.solutions.face_mesh, args)
    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "FaceLandmarker"):
        return TasksFaceLandmarkerDetector(mp, args)
    raise SystemExit("This MediaPipe install does not expose a supported face landmark API.")


def load_bbox_detector(args):
    if not args.face_detector_model:
        return None
    if not Path(args.face_detector_model).exists():
        print(f"Face detector fallback model not found, continuing without it: {args.face_detector_model}")
        return None

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("MediaPipe is not installed. Install it before running this script.") from exc

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "FaceDetector"):
        return TasksFaceDetector(mp, args)
    raise SystemExit("This MediaPipe install does not expose Tasks FaceDetector.")


def load_metadata(args, metadata_dir):
    metadata_path = input_metadata_path(args, metadata_dir)
    if metadata_path and metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if args.image_id_col not in metadata.columns:
            raise ValueError(f"Missing image ID column: {args.image_id_col}")
        if "view" not in metadata.columns:
            raise ValueError(f"Metadata must contain a 'view' column: {metadata_path}")
        return metadata[metadata["view"].astype(str).str.lower() == args.view].copy(), metadata_path
    return metadata_from_data_paths(args.input, args.view), None


def main():
    args = parse_args()

    output_dir = Path(args.output)
    metadata_dir = Path(args.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, source_metadata_path = load_metadata(args, metadata_dir)
    dataset_root = infer_dataset_root(args.input, metadata_dir)
    output_metadata = output_metadata_path(args, metadata_dir)
    skipped_metadata = skipped_metadata_path(args, metadata_dir)

    rows = []
    skipped = []
    detector = load_face_detector(args)
    bbox_detector = load_bbox_detector(args)

    try:
        iterable = metadata.head(args.max_images) if args.max_images > 0 else metadata
        total = len(iterable)
        for _, row in iterable.iterrows():
            out_row = row.to_dict()
            image_id = str(row[args.image_id_col]).strip()
            aligned_filename = f"{image_id}_aligned.jpg"
            aligned_path = output_dir / aligned_filename

            out_row["view"] = args.view
            out_row["alignment_method"] = args.alignment_method
            out_row["alignment_error"] = ""
            out_row["profile_rotation_degrees"] = np.nan
            out_row["profile_flipped"] = False
            out_row["profile_direction_inferred"] = "unknown"
            out_row["profile_direction_source"] = "unknown"
            out_row["profile_direction_confidence"] = "unknown"
            out_row["profile_geometry_source"] = "not_applicable"
            out_row["profile_full_landmarks_found"] = False
            out_row["profile_detector_bbox_found"] = False
            out_row["profile_roi_retry_attempted"] = False
            out_row["profile_roi_retry_succeeded"] = False
            out_row["profile_detector_center_mass_score"] = np.nan
            out_row["profile_detector_center_mass_direction"] = "unknown"
            out_row["profile_detector_center_mass_confidence"] = "unknown"
            out_row["profile_detector_edge_balance_score"] = np.nan
            out_row["profile_detector_edge_balance_direction"] = "unknown"
            out_row["profile_detector_edge_balance_confidence"] = "unknown"

            if aligned_path.exists() and not args.overwrite:
                out_row["is_aligned"] = True
                out_row["aligned_filename"] = aligned_filename
                out_row["aligned_path"] = relative_or_absolute(aligned_path)
                rows.append(out_row)
                continue

            image_path = resolve_image_path(row, args, dataset_root)
            if image_path is None:
                reason = "raw_image_not_found"
                out_row["is_aligned"] = False
                out_row["aligned_filename"] = ""
                out_row["aligned_path"] = ""
                out_row["alignment_error"] = reason
                skipped.append({"image_id": image_id, "reason": reason, "source_path": row.get(args.image_path_col, "")})
                rows.append(out_row)
                continue

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                reason = "image_read_failed"
                out_row["is_aligned"] = False
                out_row["aligned_filename"] = ""
                out_row["aligned_path"] = ""
                out_row["alignment_error"] = reason
                skipped.append({"image_id": image_id, "reason": reason, "source_path": str(image_path)})
                rows.append(out_row)
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_bgr.shape[:2]
            face_landmarks = detector.detect(image_rgb)

            if args.view == "profile":
                aligned = crop_profile_image_v2(
                    image_bgr,
                    image_rgb,
                    detector,
                    bbox_detector,
                    face_landmarks,
                    args,
                    out_row,
                )
            elif face_landmarks is not None:
                landmarks = landmarks_to_points(face_landmarks, width, height)
                aligned = align_image(image_bgr, landmarks, args.output_size)
                out_row["crop_mode"] = "arcface_5point"
            else:
                aligned = None
                out_row["crop_mode"] = "landmarks_not_found"

            if aligned is None:
                reason = out_row.get("crop_mode") or ("face_landmarks_not_found" if face_landmarks is None else "alignment_transform_failed")
                out_row["is_aligned"] = False
                out_row["aligned_filename"] = ""
                out_row["aligned_path"] = ""
                out_row["alignment_error"] = reason
                skipped.append({"image_id": image_id, "reason": reason, "source_path": str(image_path)})
                rows.append(out_row)
                continue

            cv2.imwrite(str(aligned_path), aligned)
            out_row["is_aligned"] = True
            out_row["aligned_filename"] = aligned_filename
            out_row["aligned_path"] = relative_or_absolute(aligned_path)
            rows.append(out_row)

            processed = len(rows)
            if processed % 500 == 0 or processed == total:
                print(f"Processed {processed}/{total} images; skipped {len(skipped)}.")
    finally:
        detector.close()
        if bbox_detector is not None:
            bbox_detector.close()

    output = pd.DataFrame(rows)
    output.to_csv(output_metadata, index=False)
    pd.DataFrame(skipped).to_csv(skipped_metadata, index=False)

    if source_metadata_path:
        print(f"Source metadata: {source_metadata_path}")
    print(f"Saved MediaPipe metadata: {output_metadata}")
    print(f"Saved skipped alignment report: {skipped_metadata}")
    print(f"Aligned images: {int(output['is_aligned'].map(truthy).sum())}")
    print(f"Skipped images: {len(skipped)}")


if __name__ == "__main__":
    main()
