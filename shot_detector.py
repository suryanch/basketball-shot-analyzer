from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import math

import config
from angle_calculator import calculate_vector_angle


@dataclass
class ShotEvent:
    frame: int
    timestamp: float
    elbow_angle_at_set: Optional[float]
    knee_angle_at_initiation: Optional[float]
    release_arc: Optional[float]       # wrist-based arc
    trajectory: list                   # list of (x, y) wrist positions during shot
    ball_release_arc: Optional[float] = None  # ball-based arc (preferred when available)
    elbow_angle_at_release: Optional[float] = None  # elbow angle at ball-release frame
    knee_angle_at_release: Optional[float] = None   # knee angle at ball-release frame
    release_frame_index: Optional[int] = None       # frame index when ball left hand
    elbow_angle_at_load: Optional[float] = None     # elbow at deepest squat (pre-jump)
    knee_angle_at_load: Optional[float] = None      # knee at deepest squat (pre-jump)
    load_frame_index: Optional[int] = None          # frame index of loading position

    @property
    def effective_release_arc(self) -> Optional[float]:
        return self.ball_release_arc if self.ball_release_arc is not None else self.release_arc


class ShotDetector:
    IDLE = "IDLE"
    COCKING = "COCKING"
    RELEASE = "RELEASE"
    FOLLOW_THROUGH = "FOLLOW_THROUGH"

    def __init__(self):
        self.state = self.IDLE
        self.wrist_history: deque = deque(maxlen=config.WRIST_TRAJECTORY_WINDOW)
        self.shot_trajectory: list = []
        self.shot_events: list = []
        self.cooldown_frames = 0

        # Tracking for release detection
        self._upward_count = 0
        self._cocking_elbow_angle: Optional[float] = None
        self._cocking_knee_angle: Optional[float] = None
        self._peak_wrist_speed = 0.0
        self._shot_start_frame = 0
        self._shot_start_ts = 0.0
        self._release_elbow_angle: Optional[float] = None
        self._release_knee_angle: Optional[float] = None
        self._release_frame_index: Optional[int] = None
        self._min_knee_angle: float = float('inf')
        self._min_knee_elbow: Optional[float] = None
        self._min_knee_frame_idx: Optional[int] = None
        self._load_elbow: Optional[float] = None
        self._load_knee: Optional[float] = None
        self._load_frame_idx: Optional[int] = None

    def update(self, frame_idx: int, timestamp: float,
               wrist_pos: Optional[tuple],
               knee_angle: Optional[float],
               elbow_angle: Optional[float],
               ball_state=None) -> tuple:
        """Returns (phase_str, release_arc | None, load_event | None).
        load_event is a dict {frame_idx, elbow, knee} emitted once at COCKING→RELEASE."""
        release_arc = None
        load_event = None

        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return self.state, None, None

        # Capture angles at the exact frame the ball leaves the hand
        if ball_state is not None and ball_state.just_released:
            self._release_elbow_angle = elbow_angle
            self._release_knee_angle = knee_angle
            self._release_frame_index = frame_idx

        if wrist_pos is not None:
            self.wrist_history.append(wrist_pos)
        else:
            self.wrist_history.append(None)

        # Compute wrist vertical velocity (image coords: negative = upward)
        vy = self._compute_vy()
        speed = abs(vy) if vy is not None else 0.0

        if self.state == self.IDLE:
            if self._is_cocking(knee_angle, vy):
                self.state = self.COCKING
                self._cocking_elbow_angle = elbow_angle
                self._cocking_knee_angle = knee_angle
                self._shot_start_frame = frame_idx
                self._shot_start_ts = timestamp
                self.shot_trajectory = list(self.wrist_history)
                self._upward_count = 0
                self._peak_wrist_speed = speed

        elif self.state == self.COCKING:
            if wrist_pos:
                self.shot_trajectory.append(wrist_pos)
            self._peak_wrist_speed = max(self._peak_wrist_speed, speed)

            # Track the frame with maximum knee bend (minimum angle)
            if knee_angle is not None and knee_angle < self._min_knee_angle:
                self._min_knee_angle = knee_angle
                self._min_knee_elbow = elbow_angle
                self._min_knee_frame_idx = frame_idx

            if vy is not None and vy < config.WRIST_UPWARD_THRESHOLD:
                self._upward_count += 1
            else:
                self._upward_count = max(0, self._upward_count - 1)

            if self._upward_count >= config.RELEASE_CONSECUTIVE_FRAMES:
                self.state = self.RELEASE
                release_arc = self._compute_release_arc()
                # Emit load_event and preserve values for ShotEvent
                if self._min_knee_frame_idx is not None:
                    load_event = {
                        "frame_idx": self._min_knee_frame_idx,
                        "elbow": self._min_knee_elbow,
                        "knee": self._min_knee_angle,
                    }
                    self._load_elbow = self._min_knee_elbow
                    self._load_knee = self._min_knee_angle
                    self._load_frame_idx = self._min_knee_frame_idx
                # Reset min tracking for next shot
                self._min_knee_angle = float('inf')
                self._min_knee_elbow = None
                self._min_knee_frame_idx = None

        elif self.state == self.RELEASE:
            if wrist_pos:
                self.shot_trajectory.append(wrist_pos)
            if speed < self._peak_wrist_speed * config.FOLLOW_THROUGH_VELOCITY_DROP:
                self.state = self.FOLLOW_THROUGH
                arc = self._compute_release_arc()
                ball_arc = (ball_state.release_arc
                            if ball_state and ball_state.release_arc is not None
                            else None)
                event = ShotEvent(
                    frame=self._shot_start_frame,
                    timestamp=self._shot_start_ts,
                    elbow_angle_at_set=self._cocking_elbow_angle,
                    knee_angle_at_initiation=self._cocking_knee_angle,
                    release_arc=arc,
                    trajectory=list(self.shot_trajectory),
                    ball_release_arc=ball_arc,
                    elbow_angle_at_release=self._release_elbow_angle,
                    knee_angle_at_release=self._release_knee_angle,
                    release_frame_index=self._release_frame_index,
                    elbow_angle_at_load=self._load_elbow,
                    knee_angle_at_load=self._load_knee,
                    load_frame_index=self._load_frame_idx,
                )
                self._release_elbow_angle = None
                self._release_knee_angle = None
                self._release_frame_index = None
                self._load_elbow = None
                self._load_knee = None
                self._load_frame_idx = None
                self.shot_events.append(event)

        elif self.state == self.FOLLOW_THROUGH:
            self.state = self.IDLE
            self.cooldown_frames = config.SHOT_COOLDOWN_FRAMES
            self.shot_trajectory = []
            self._upward_count = 0
            self._peak_wrist_speed = 0.0
            self._min_knee_angle = float('inf')
            self._min_knee_elbow = None
            self._min_knee_frame_idx = None

        return self.state, release_arc, load_event

    def _is_cocking(self, knee_angle: Optional[float], vy: Optional[float]) -> bool:
        knee_bent = knee_angle is not None and knee_angle < config.COCKING_KNEE_THRESHOLD
        wrist_loading = vy is not None and vy > 5.0  # wrist moving downward (positive y)
        return knee_bent and wrist_loading

    def _compute_vy(self) -> Optional[float]:
        """Compute vertical velocity from last two valid wrist positions."""
        valid = [p for p in self.wrist_history if p is not None]
        if len(valid) < 2:
            return None
        return valid[-1][1] - valid[-2][1]  # positive = downward in image

    def _compute_release_arc(self) -> Optional[float]:
        """Compute release arc angle from wrist trajectory (degrees above horizontal)."""
        valid = [p for p in self.shot_trajectory if p is not None]
        if len(valid) < 3:
            return None
        # Use the last few positions to estimate upward direction vector
        n = min(5, len(valid))
        recent = valid[-n:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return None
        # Convert to math coords (invert y), angle from horizontal
        math_dy = -dy
        angle = math.degrees(math.atan2(math_dy, abs(dx) if abs(dx) > 1e-3 else 0.01))
        return max(0.0, angle)

    def get_shot_summary(self) -> list:
        return [
            {
                "frame": e.frame,
                "timestamp": round(e.timestamp, 3),
                "elbow_angle_at_set": round(e.elbow_angle_at_set, 1) if e.elbow_angle_at_set else None,
                "knee_angle_at_initiation": round(e.knee_angle_at_initiation, 1) if e.knee_angle_at_initiation else None,
                "release_arc": round(e.effective_release_arc, 1) if e.effective_release_arc else None,
                "release_arc_source": "ball" if e.ball_release_arc is not None else "wrist",
                "trajectory_length": len(e.trajectory),
            }
            for e in self.shot_events
        ]
