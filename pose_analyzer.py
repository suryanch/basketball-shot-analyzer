from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ultralytics import YOLO
import config

import config
from angle_calculator import calculate_angle
from ball_tracker import BallTracker, BallState


@dataclass
class KeypointData:
    keypoints: np.ndarray    # shape (17, 2)
    confidences: np.ndarray  # shape (17,)
    bbox: tuple              # (x1, y1, x2, y2)
    person_id: int = 0


@dataclass
class HeldObject:
    bbox: tuple          # (x1, y1, x2, y2)
    label: str
    confidence: float
    is_round: bool       # True → draw ellipse nodes; False → draw rect corner nodes
    hand_landmarks: list = None   # 21 (x,y) pixel coords if available


@dataclass
class FrameAnalysis:
    frame_index: int
    timestamp: float
    persons: list
    elbow_angle_left: Optional[float] = None
    elbow_angle_right: Optional[float] = None
    knee_angle_left: Optional[float] = None
    knee_angle_right: Optional[float] = None
    wrist_pos: Optional[tuple] = None       # shooting wrist (x, y)
    shot_phase: str = "IDLE"
    release_arc: Optional[float] = None
    annotated_frame: Optional[np.ndarray] = None
    shooting_side: str = "right"            # "left" or "right"
    held_objects: list = None               # list of HeldObject
    hand_landmarks: list = None             # list of landmark lists (one per hand, always)
    ball_state: Optional[BallState] = None


# COCO classes considered "round" for node fitting
_ROUND_CLASSES = {"sports ball", "orange", "apple", "clock", "frisbee"}


class PoseAnalyzer:
    def __init__(self, model_path: str = config.DEFAULT_MODEL,
                 conf: float = config.MIN_CONFIDENCE,
                 iou: float = config.DEFAULT_IOU,
                 hand_mode: str = "crop"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.hand_mode = hand_mode

        from hand_detector import HandObjectDetector
        self._hand_detector = HandObjectDetector()
        self._ball_tracker = BallTracker()
        self.obj_model = None

    def analyze_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> FrameAnalysis:
        timestamp = frame_idx / fps if fps > 0 else 0.0
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False, device=config.INFERENCE_DEVICE)
        persons = self._extract_keypoints(results)
        primary = self._select_primary_person(persons)

        analysis = FrameAnalysis(
            frame_index=frame_idx,
            timestamp=timestamp,
            persons=persons,
            annotated_frame=None,
            held_objects=[],
        )

        if primary is not None:
            angles = self._compute_angles(primary)
            analysis.elbow_angle_left = angles.get("elbow_left")
            analysis.elbow_angle_right = angles.get("elbow_right")
            analysis.knee_angle_left = angles.get("knee_left")
            analysis.knee_angle_right = angles.get("knee_right")

            # Determine shooting side and wrist position
            shooting_side = self._detect_shooting_side(primary)
            analysis.shooting_side = shooting_side
            if shooting_side == "right":
                wrist_idx = config.KP_RIGHT_WRIST
            else:
                wrist_idx = config.KP_LEFT_WRIST

            if primary.confidences[wrist_idx] >= config.MIN_CONFIDENCE:
                wx, wy = primary.keypoints[wrist_idx]
                analysis.wrist_pos = (float(wx), float(wy))

        # Detect held objects and always-on hand landmarks
        analysis.held_objects, analysis.hand_landmarks = self._detect_held_objects_hand(frame)

        # If an object is held, use its center as the tracking position instead of the wrist
        if analysis.held_objects:
            obj = analysis.held_objects[0]
            x1, y1, x2, y2 = obj.bbox
            analysis.wrist_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Ball tracking (full frame) — pass person bbox to filter background objects
        person_bbox = primary.bbox if primary is not None else None
        analysis.ball_state = self._ball_tracker.update(frame, analysis.wrist_pos, person_bbox)

        return analysis

    def _detect_held_objects_hand(self, frame: np.ndarray) -> tuple:
        """Convert HandResult detections into HeldObject list + always-on landmarks."""
        hand_results = self._hand_detector.detect(frame)
        held = []
        all_landmarks = []

        for hr in hand_results:
            # Always collect landmarks regardless of holding state
            if hr.landmarks_px:
                all_landmarks.append(hr.landmarks_px)

            if not hr.is_holding:
                continue
            bbox = hr.held_bbox if hr.held_bbox is not None else hr.hand_bbox
            label = hr.held_label or "held"
            conf = hr.held_conf or 1.0
            is_round = hr.held_is_round
            held.append(HeldObject(
                bbox=bbox,
                label=label,
                confidence=conf,
                is_round=is_round,
                hand_landmarks=hr.landmarks_px,
            ))

        return held, all_landmarks

    def _detect_held_objects(self, frame: np.ndarray, wrist_positions: list) -> list:
        """Detect held objects by cropping to each wrist region, with sticky persistence."""
        h, w = frame.shape[:2]
        pad = config.HELD_OBJECT_WRIST_RADIUS
        fresh = []

        for wx, wy in wrist_positions:
            # Crop a region around the wrist
            cx1 = max(0, int(wx - pad))
            cy1 = max(0, int(wy - pad))
            cx2 = min(w, int(wx + pad))
            cy2 = min(h, int(wy + pad))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = frame[cy1:cy2, cx1:cx2]

            obj_results = self.obj_model(crop, conf=0.15, iou=self.iou, verbose=False)
            if not obj_results or obj_results[0].boxes is None:
                continue

            boxes = obj_results[0].boxes
            names = obj_results[0].names

            # Find the single highest-confidence whitelisted detection in this crop
            best = None
            for i in range(len(boxes)):
                label = names[int(boxes.cls[i].item())]
                if label not in config.HELD_OBJECT_ALLOWED_CLASSES:
                    continue
                conf = float(boxes.conf[i].item())
                if best is None or conf > best[1]:
                    bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy()
                    best = (label, conf, bx1, by1, bx2, by2)

            if best is not None:
                label, conf, bx1, by1, bx2, by2 = best
                fx1, fy1 = bx1 + cx1, by1 + cy1
                fx2, fy2 = bx2 + cx1, by2 + cy1
                fresh.append(HeldObject(
                    bbox=(float(fx1), float(fy1), float(fx2), float(fy2)),
                    label=label,
                    confidence=conf,
                    is_round=label in _ROUND_CLASSES,
                ))

        if fresh:
            # New detection — reset sticky
            self._sticky_objects = fresh
            self._sticky_ttl = self._STICKY_FRAMES
            return fresh

        # No fresh detection — count down sticky
        if self._sticky_ttl > 0:
            self._sticky_ttl -= 1
            return self._sticky_objects

        self._sticky_objects = []
        return []

    def _extract_keypoints(self, results) -> list:
        persons = []
        if not results or results[0].keypoints is None:
            return persons
        result = results[0]
        kps_xy = result.keypoints.xy.cpu().numpy()     # (N, 17, 2)
        kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None

        for i in range(len(kps_xy)):
            kp = kps_xy[i]
            conf = kps_conf[i] if kps_conf is not None else np.ones(17)
            bbox = tuple(boxes[i]) if boxes is not None and i < len(boxes) else (0, 0, 0, 0)
            persons.append(KeypointData(keypoints=kp, confidences=conf, bbox=bbox, person_id=i))

        return persons

    def _select_primary_person(self, persons: list) -> Optional[KeypointData]:
        if not persons:
            return None
        # Pick person with largest bounding box area
        def bbox_area(p):
            x1, y1, x2, y2 = p.bbox
            return max(0, x2 - x1) * max(0, y2 - y1)
        return max(persons, key=bbox_area)

    def _detect_shooting_side(self, kp: KeypointData) -> str:
        """Determine shooting side by which wrist is higher (lower y in image)."""
        lw_conf = kp.confidences[config.KP_LEFT_WRIST]
        rw_conf = kp.confidences[config.KP_RIGHT_WRIST]
        if lw_conf < config.MIN_CONFIDENCE and rw_conf < config.MIN_CONFIDENCE:
            return "right"
        if lw_conf < config.MIN_CONFIDENCE:
            return "right"
        if rw_conf < config.MIN_CONFIDENCE:
            return "left"
        # Higher wrist (smaller y) = shooting wrist
        lw_y = kp.keypoints[config.KP_LEFT_WRIST][1]
        rw_y = kp.keypoints[config.KP_RIGHT_WRIST][1]
        return "left" if lw_y < rw_y else "right"

    def _compute_angles(self, kp: KeypointData) -> dict:
        angles = {}

        def get_point(idx):
            if kp.confidences[idx] >= config.MIN_CONFIDENCE:
                return tuple(kp.keypoints[idx].tolist())
            return None

        # Elbow angles (shoulder - elbow - wrist)
        angles["elbow_left"] = calculate_angle(
            get_point(config.KP_LEFT_SHOULDER),
            get_point(config.KP_LEFT_ELBOW),
            get_point(config.KP_LEFT_WRIST),
        )
        angles["elbow_right"] = calculate_angle(
            get_point(config.KP_RIGHT_SHOULDER),
            get_point(config.KP_RIGHT_ELBOW),
            get_point(config.KP_RIGHT_WRIST),
        )

        # Knee angles (hip - knee - ankle)
        angles["knee_left"] = calculate_angle(
            get_point(config.KP_LEFT_HIP),
            get_point(config.KP_LEFT_KNEE),
            get_point(config.KP_LEFT_ANKLE),
        )
        angles["knee_right"] = calculate_angle(
            get_point(config.KP_RIGHT_HIP),
            get_point(config.KP_RIGHT_KNEE),
            get_point(config.KP_RIGHT_ANKLE),
        )

        return angles
