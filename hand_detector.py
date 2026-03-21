"""
Hand detection modes:
  HandObjectDetector   — Option 2: MediaPipe locates the hand precisely,
                         then YOLO classifies what's being held.
  MediaPipeHandTracker — Option 3: Pure MediaPipe 21-landmark tracking;
                         grip inferred from finger curl, no YOLO involved.
"""
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config

# Auto-download the hand landmark model on first use
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("[INFO] Downloading hand_landmarker.task model (~10 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] hand_landmarker.task downloaded.")

# MediaPipe hand connections for drawing the 21-landmark skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

_FINGERTIP_IDS = [8, 12, 16, 20]
_MCP_IDS       = [5,  9, 13, 17]
_PALM_IDS      = [0, 5, 9, 13, 17]  # wrist + base knuckles = palm center
_ROUND_CLASSES = {"sports ball", "orange", "apple", "clock", "frisbee"}

# Per-class expected aspect ratios (height / width) for bbox expansion.
# Used when a detection touches the crop edge (partial object seen).
_CLASS_ASPECT = {
    "cell phone":   2.0,   # tall
    "remote":       3.0,   # very tall
    "bottle":       3.5,   # tall
    "cup":          1.2,
    "sports ball":  1.0,   # square
    "tennis racket":2.5,
    "baseball bat": 6.0,
    "frisbee":      1.0,
    "orange":       1.0,
    "apple":        1.0,
    "banana":       2.5,
    "scissors":     2.0,
    "knife":        4.0,
    "fork":         3.0,
    "spoon":        3.0,
}
_EDGE_MARGIN = 10  # px — how close to crop edge counts as "touching"


@dataclass
class HandResult:
    landmarks_px: list          # 21 (x, y) pixel coords
    hand_bbox: tuple            # (x1, y1, x2, y2)
    is_holding: bool
    held_label: Optional[str] = None
    held_conf: Optional[float] = None
    held_bbox: Optional[tuple] = None
    held_is_round: bool = False


def _build_landmarker(min_det: float = 0.4, min_track: float = 0.4):
    _ensure_model()
    base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,
        min_hand_detection_confidence=min_det,
        min_hand_presence_confidence=min_det,
        min_tracking_confidence=min_track,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _landmarks_to_px(hand_landmarks, w: int, h: int) -> list:
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]


def _hand_bbox(lm_px: list, w: int, h: int, pad: int = 20) -> tuple:
    """Tight hand bbox used for drawing landmarks."""
    xs = [p[0] for p in lm_px]
    ys = [p[1] for p in lm_px]
    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(w, max(xs) + pad),
        min(h, max(ys) + pad),
    )


def _palm_centric_crop(lm_px: list, w: int, h: int) -> tuple:
    """Step 1+2: Large square crop centered on palm, sized 2x hand + 150px padding."""
    palm_pts = [lm_px[i] for i in _PALM_IDS]
    cx = int(sum(p[0] for p in palm_pts) / len(palm_pts))
    cy = int(sum(p[1] for p in palm_pts) / len(palm_pts))

    # Hand size = distance from wrist to middle fingertip
    wrist = lm_px[0]
    mid_tip = lm_px[12]
    hand_size = int(((wrist[0] - mid_tip[0])**2 + (wrist[1] - mid_tip[1])**2) ** 0.5)
    half = max(hand_size, 80) + 150  # at least 150px around palm center

    return (
        max(0, cx - half),
        max(0, cy - half),
        min(w, cx + half),
        min(h, cy + half),
    )


def _expand_bbox(bx1, by1, bx2, by2, label: str, crop_w: int, crop_h: int) -> tuple:
    """Step 3: If bbox touches crop edge, expand to estimated full object size."""
    touches_edge = (
        bx1 <= _EDGE_MARGIN or by1 <= _EDGE_MARGIN or
        bx2 >= crop_w - _EDGE_MARGIN or by2 >= crop_h - _EDGE_MARGIN
    )
    if not touches_edge:
        return bx1, by1, bx2, by2

    aspect = _CLASS_ASPECT.get(label, 1.5)  # height/width
    det_w = bx2 - bx1
    det_h = by2 - by1

    # Infer the larger dimension from the smaller one using expected aspect ratio
    if det_h / max(det_w, 1) < aspect * 0.7:
        # Object appears wider than tall — height is likely clipped; expand it
        full_h = det_w * aspect
        extra = (full_h - det_h) / 2
        # Expand toward whichever edge the bbox is touching
        if by1 <= _EDGE_MARGIN:
            by1 = max(0, by1 - extra)
        if by2 >= crop_h - _EDGE_MARGIN:
            by2 = min(crop_h, by2 + extra)
    elif det_w / max(det_h, 1) < (1.0 / aspect) * 0.7:
        # Object appears taller than wide — width is likely clipped; expand it
        full_w = det_h / aspect
        extra = (full_w - det_w) / 2
        if bx1 <= _EDGE_MARGIN:
            bx1 = max(0, bx1 - extra)
        if bx2 >= crop_w - _EDGE_MARGIN:
            bx2 = min(crop_w, bx2 + extra)

    return bx1, by1, bx2, by2


# ─── Option 2 ────────────────────────────────────────────────────────────────

class HandObjectDetector:
    """MediaPipe hand bbox → YOLO object detection inside that region."""

    def __init__(self):
        self._landmarker = _build_landmarker(min_det=0.4, min_track=0.4)
        from ultralytics import YOLO
        self._obj_model = YOLO(config.HELD_OBJECT_MODEL)
        self._sticky: list = []
        self._sticky_ttl = 0
        self._STICKY_FRAMES = 8
        self._frame_ts = 0  # monotonic ms timestamp for Tasks API

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33  # ~30 fps increment
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.hand_landmarks:
            if self._sticky_ttl > 0:
                self._sticky_ttl -= 1
                return self._sticky
            return []

        results = []
        for hand_lms in result.hand_landmarks:
            lm_px = _landmarks_to_px(hand_lms, w, h)
            draw_bbox = _hand_bbox(lm_px, w, h, pad=20)

            # Step 1+2: palm-centric large crop so full object is visible to YOLO
            x1, y1, x2, y2 = _palm_centric_crop(lm_px, w, h)
            crop_w, crop_h = x2 - x1, y2 - y1

            held_label, held_conf, held_bbox, held_is_round = None, None, None, False
            is_holding = False

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                obj_res = self._obj_model(crop, conf=0.2, verbose=False)
                if obj_res and obj_res[0].boxes is not None:
                    best = None
                    names = obj_res[0].names
                    for i in range(len(obj_res[0].boxes)):
                        label = names[int(obj_res[0].boxes.cls[i].item())]
                        if label not in config.HELD_OBJECT_ALLOWED_CLASSES:
                            continue
                        conf = float(obj_res[0].boxes.conf[i].item())
                        if best is None or conf > best[1]:
                            bx1, by1, bx2, by2 = obj_res[0].boxes.xyxy[i].cpu().numpy()
                            best = (label, conf, bx1, by1, bx2, by2)
                    if best:
                        held_label, held_conf = best[0], best[1]
                        # Step 3: expand partial bbox if it touches crop edge
                        bx1, by1, bx2, by2 = _expand_bbox(
                            best[2], best[3], best[4], best[5], held_label, crop_w, crop_h
                        )
                        # Map back to full frame coords
                        held_bbox = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                        held_is_round = held_label in _ROUND_CLASSES
                        is_holding = True

            results.append(HandResult(
                landmarks_px=lm_px,
                hand_bbox=draw_bbox,
                is_holding=is_holding,
                held_label=held_label,
                held_conf=held_conf,
                held_bbox=held_bbox,
                held_is_round=held_is_round,
            ))

        if results:
            self._sticky = results
            self._sticky_ttl = self._STICKY_FRAMES
        elif self._sticky_ttl > 0:
            self._sticky_ttl -= 1
            return self._sticky

        return results


# ─── Option 3 ────────────────────────────────────────────────────────────────

class MediaPipeHandTracker:
    """Pure MediaPipe 21-landmark tracking — grip inferred from finger curl."""

    def __init__(self):
        self._landmarker = _build_landmarker(min_det=0.5, min_track=0.5)
        self._sticky: list = []
        self._sticky_ttl = 0
        self._STICKY_FRAMES = 5
        self._frame_ts = 0

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.hand_landmarks:
            if self._sticky_ttl > 0:
                self._sticky_ttl -= 1
                return self._sticky
            return []

        results = []
        for hand_lms in result.hand_landmarks:
            lm_px = _landmarks_to_px(hand_lms, w, h)
            x1, y1, x2, y2 = _hand_bbox(lm_px, w, h, pad=20)
            is_holding = self._compute_grip(hand_lms)
            results.append(HandResult(
                landmarks_px=lm_px,
                hand_bbox=(x1, y1, x2, y2),
                is_holding=is_holding,
            ))

        if results:
            self._sticky = results
            self._sticky_ttl = self._STICKY_FRAMES
        elif self._sticky_ttl > 0:
            self._sticky_ttl -= 1
            return self._sticky

        return results

    def _compute_grip(self, hand_lms) -> bool:
        wrist_y = hand_lms[0].y
        curl_count = 0
        for tip_id, mcp_id in zip(_FINGERTIP_IDS, _MCP_IDS):
            tip_y = hand_lms[tip_id].y
            mcp_y = hand_lms[mcp_id].y
            if wrist_y > mcp_y:
                if tip_y > mcp_y:
                    curl_count += 1
            else:
                if tip_y < mcp_y:
                    curl_count += 1
        return curl_count >= 3
