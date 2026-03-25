"""
Full-frame basketball tracker.
Detects the ball on every frame, determines if it is HELD (near the hand)
or IN_FLIGHT (moving freely), and computes the release arc from the ball's
actual trajectory once it leaves the hand.
"""
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class BallState:
    detected: bool = False
    center: Optional[tuple] = None      # (cx, cy) pixel coords
    bbox: Optional[tuple] = None        # (x1, y1, x2, y2)
    state: str = "UNKNOWN"              # "HELD" | "IN_FLIGHT" | "UNKNOWN"
    release_arc: Optional[float] = None # degrees above horizontal (set once per throw)
    flight_trajectory: list = field(default_factory=list)
    just_released: bool = False         # True on the single frame of HELD→IN_FLIGHT


class BallTracker:
    def __init__(self):
        self._model = YOLO(config.HELD_OBJECT_MODEL)
        self._state = "UNKNOWN"
        self._flight_positions: list = []
        self._release_arc_cache: Optional[float] = None

    def update(self, frame: np.ndarray,
               wrist_pos: Optional[tuple],
               person_bbox: Optional[tuple] = None) -> BallState:
        # Full-frame detection filtered to sports ball
        results = self._model(
            frame,
            conf=config.BALL_CONF_THRESHOLD,
            classes=[config.BALL_COCO_CLASS_ID],
            verbose=False,
            device=config.INFERENCE_DEVICE,
        )

        ball_center = None
        ball_bbox = None

        # Expand person bbox for proximity check
        person_region = None
        if person_bbox is not None:
            pad = config.BALL_PERSON_PADDING_PX
            px1, py1, px2, py2 = person_bbox
            person_region = (px1 - pad, py1 - pad, px2 + pad, py2 + pad)

        if results and results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes
            best_conf = -1.0
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                conf = float(boxes.conf[i].item())

                # Filter: minimum size
                if w < config.BALL_MIN_SIZE_PX or h < config.BALL_MIN_SIZE_PX:
                    continue

                # Filter: must be within person region
                if person_region is not None:
                    rx1, ry1, rx2, ry2 = person_region
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue

                if conf > best_conf:
                    best_conf = conf
                    ball_center = (cx, cy)
                    ball_bbox = (float(x1), float(y1), float(x2), float(y2))

        if ball_center is None:
            return BallState(detected=False, state="UNKNOWN",
                             release_arc=self._release_arc_cache,
                             flight_trajectory=list(self._flight_positions[-30:]),
                             just_released=False)

        # Determine HELD vs IN_FLIGHT
        near_hand = False
        if wrist_pos is not None:
            dist = math.sqrt((ball_center[0] - wrist_pos[0]) ** 2 +
                             (ball_center[1] - wrist_pos[1]) ** 2)
            near_hand = dist < config.BALL_HELD_DISTANCE_PX

        prev_state = self._state
        just_released = False

        if near_hand:
            if prev_state == "IN_FLIGHT":
                self._flight_positions = []
                self._release_arc_cache = None
            self._state = "HELD"
        else:
            if prev_state == "HELD":
                # Ball just became IN_FLIGHT — player released it
                just_released = True
                self._flight_positions = [ball_center]
                self._release_arc_cache = None
            elif prev_state == "UNKNOWN":
                # First detection not near hand — already in flight, no release event
                self._flight_positions = [ball_center]
                self._release_arc_cache = None
            else:
                self._flight_positions.append(ball_center)
            self._state = "IN_FLIGHT"

            if (self._release_arc_cache is None and
                    len(self._flight_positions) >= config.BALL_MIN_FLIGHT_FRAMES):
                self._release_arc_cache = self._compute_arc()

        return BallState(
            detected=True,
            center=ball_center,
            bbox=ball_bbox,
            state=self._state,
            release_arc=self._release_arc_cache,
            flight_trajectory=list(self._flight_positions[-30:]),
            just_released=just_released,
        )

    def _compute_arc(self) -> Optional[float]:
        n = min(config.BALL_ARC_SMOOTHING_WINDOW, len(self._flight_positions))
        pts = self._flight_positions[:n]
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if abs(dx) < 1 and abs(dy) < 1:
            return None
        math_dy = -dy  # invert: image y increases downward
        angle = math.degrees(math.atan2(math_dy, abs(dx) if abs(dx) > 0.01 else 0.01))
        return max(0.0, angle)
