import cv2
import numpy as np
import math
from typing import Optional
from PIL import ImageFont, ImageDraw, Image

import config
from pose_analyzer import FrameAnalysis, KeypointData, HeldObject

# Load a font that supports the degree symbol; fall back to default if unavailable
try:
    _ANGLE_FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
except Exception:
    try:
        _ANGLE_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        _ANGLE_FONT = ImageFont.load_default()


def _put_unicode_text(frame: np.ndarray, text: str, pos: tuple, color_bgr: tuple) -> np.ndarray:
    """Draw Unicode text (e.g. with °) onto an OpenCV frame using Pillow."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # PIL uses RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=_ANGLE_FONT, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Color constants (BGR)
COLOR_SKELETON = (0, 255, 0)
COLOR_JOINT = (0, 200, 255)
COLOR_ANGLE_TEXT = (255, 255, 0)
COLOR_TRAJECTORY = (255, 165, 0)
COLOR_BALL_TRAJECTORY = (0, 255, 100)

PHASE_COLORS = {
    "IDLE": (150, 150, 150),
    "COCKING": (0, 215, 255),
    "RELEASE": (0, 0, 255),
    "FOLLOW_THROUGH": (0, 255, 0),
}


def draw_skeleton(frame: np.ndarray, kp_data: KeypointData) -> np.ndarray:
    kps = kp_data.keypoints
    confs = kp_data.confidences

    for (i, j) in config.SKELETON_CONNECTIONS:
        if confs[i] >= config.MIN_CONFIDENCE and confs[j] >= config.MIN_CONFIDENCE:
            pt1 = (int(kps[i][0]), int(kps[i][1]))
            pt2 = (int(kps[j][0]), int(kps[j][1]))
            cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2)

    # Skip face keypoints (nose, eyes, ears — indices 0-4)
    for i in range(5, 17):
        if confs[i] >= config.MIN_CONFIDENCE:
            pt = (int(kps[i][0]), int(kps[i][1]))
            cv2.circle(frame, pt, 4, COLOR_JOINT, -1)

    return frame


def draw_angles(frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
    if not analysis.persons:
        return frame

    primary = analysis.persons[0]
    kps = primary.keypoints
    confs = primary.confidences

    def put_angle(idx, label, angle):
        if angle is None or math.isnan(angle):
            return
        if confs[idx] >= config.MIN_CONFIDENCE:
            pt = (int(kps[idx][0]) + 8, int(kps[idx][1]) - 28)
            nonlocal frame
            frame = _put_unicode_text(frame, f"{angle:.0f}°", pt, COLOR_ANGLE_TEXT)

    # Only draw angles for the shooting side to avoid overlapping labels
    if analysis.shooting_side == "right":
        put_angle(config.KP_RIGHT_ELBOW, "RE", analysis.elbow_angle_right)
        put_angle(config.KP_RIGHT_KNEE, "RK", analysis.knee_angle_right)
    else:
        put_angle(config.KP_LEFT_ELBOW, "LE", analysis.elbow_angle_left)
        put_angle(config.KP_LEFT_KNEE, "LK", analysis.knee_angle_left)

    # Release arc in top-right corner
    h, w = frame.shape[:2]
    y_offset = 10
    if analysis.release_arc is not None:
        frame = _put_unicode_text(frame, f"Arc(wrist): {analysis.release_arc:.1f}°",
                                  (w - 230, y_offset), (0, 255, 255))
        y_offset += 34
    ball_arc = (analysis.ball_state.release_arc
                if analysis.ball_state and analysis.ball_state.release_arc is not None
                else None)
    if ball_arc is not None:
        frame = _put_unicode_text(frame, f"Arc(ball): {ball_arc:.1f}°",
                                  (w - 230, y_offset), COLOR_BALL_TRAJECTORY)

    return frame


def draw_shot_phase(frame: np.ndarray, phase: str,
                    frame_idx: int, timestamp: float) -> np.ndarray:
    color = PHASE_COLORS.get(phase, (255, 255, 255))
    text = f"Phase: {phase}  |  Frame: {frame_idx}  |  {timestamp:.2f}s"
    cv2.rectangle(frame, (0, 0), (len(text) * 11, 35), (0, 0, 0), -1)
    cv2.putText(frame, text, (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame


def draw_wrist_trajectory(frame: np.ndarray, trajectory: list) -> np.ndarray:
    """Draw fading polyline of last 20 wrist positions."""
    recent = [p for p in trajectory[-20:] if p is not None]
    if len(recent) < 2:
        return frame
    for i in range(1, len(recent)):
        alpha = i / len(recent)
        color = (
            int(COLOR_TRAJECTORY[0] * alpha),
            int(COLOR_TRAJECTORY[1] * alpha),
            int(COLOR_TRAJECTORY[2] * alpha),
        )
        pt1 = (int(recent[i - 1][0]), int(recent[i - 1][1]))
        pt2 = (int(recent[i][0]), int(recent[i][1]))
        cv2.line(frame, pt1, pt2, color, 3)
    return frame


_MP_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand_landmarks(frame: np.ndarray, landmarks_px: list) -> np.ndarray:
    """Draw MediaPipe 21-point hand skeleton."""
    for (i, j) in _MP_HAND_CONNECTIONS:
        if i < len(landmarks_px) and j < len(landmarks_px):
            cv2.line(frame, landmarks_px[i], landmarks_px[j], (0, 255, 180), 2)
    for pt in landmarks_px:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
    return frame


def draw_held_object_nodes(frame: np.ndarray, obj: HeldObject) -> np.ndarray:
    """Draw nodes fitted to a held object's bounding box."""
    x1, y1, x2, y2 = obj.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
    n = config.HELD_OBJECT_NUM_NODES

    if obj.is_round:
        # Draw ellipse outline + evenly spaced nodes around perimeter
        cv2.ellipse(frame, (int(cx), int(cy)), (int(rx), int(ry)),
                    0, 0, 360, config.HELD_OBJECT_LINE_COLOR, 2)
        pts = []
        for k in range(n):
            angle = 2 * math.pi * k / n
            px = int(cx + rx * math.cos(angle))
            py = int(cy + ry * math.sin(angle))
            pts.append((px, py))
            cv2.circle(frame, (px, py), 5, config.HELD_OBJECT_NODE_COLOR, -1)
    else:
        # Rectangle: draw bbox + corner + midpoint nodes
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      config.HELD_OBJECT_LINE_COLOR, 2)
        pts = [
            (int(x1), int(y1)), (int(cx), int(y1)), (int(x2), int(y1)),
            (int(x2), int(cy)),
            (int(x2), int(y2)), (int(cx), int(y2)), (int(x1), int(y2)),
            (int(x1), int(cy)),
        ]
        for pt in pts:
            cv2.circle(frame, pt, 5, config.HELD_OBJECT_NODE_COLOR, -1)

    # Label above the object
    cv2.putText(frame, f"{obj.label} {obj.confidence:.0%}",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.HELD_OBJECT_NODE_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_eye_gaze(frame: np.ndarray, gaze) -> np.ndarray:
    """Draw iris dots, gaze arrows, and angle labels for both eyes."""
    IRIS_COLOR  = (0, 255, 255)
    ARROW_COLOR = (255, 100, 0)

    def draw_eye(f, iris, center, h_ang, v_ang):
        if iris is None or center is None:
            return f
        cv2.circle(f, iris, 4, IRIS_COLOR, -1)
        arrow_len = 25
        dx = int(arrow_len * math.sin(math.radians(h_ang)))
        dy = int(arrow_len * math.sin(math.radians(v_ang)))
        tip = (iris[0] + dx, iris[1] + dy)
        cv2.arrowedLine(f, iris, tip, ARROW_COLOR, 2, tipLength=0.4)
        label = f"H:{h_ang:+.0f} V:{v_ang:+.0f}"
        return _put_unicode_text(f, label, (iris[0] + 8, iris[1] - 16), IRIS_COLOR)

    frame = draw_eye(frame, gaze.right_iris, gaze.right_eye_center, gaze.right_h_angle, gaze.right_v_angle)
    frame = draw_eye(frame, gaze.left_iris,  gaze.left_eye_center,  gaze.left_h_angle,  gaze.left_v_angle)
    return frame


def draw_ball_trajectory(frame: np.ndarray, trajectory: list) -> np.ndarray:
    """Draw fading green polyline of ball flight positions."""
    recent = [p for p in trajectory[-20:] if p is not None]
    if len(recent) < 2:
        return frame
    for i in range(1, len(recent)):
        alpha = i / len(recent)
        color = (
            int(COLOR_BALL_TRAJECTORY[0] * alpha),
            int(COLOR_BALL_TRAJECTORY[1] * alpha),
            int(COLOR_BALL_TRAJECTORY[2] * alpha),
        )
        pt1 = (int(recent[i - 1][0]), int(recent[i - 1][1]))
        pt2 = (int(recent[i][0]), int(recent[i][1]))
        cv2.line(frame, pt1, pt2, color, 3)
    # Circle at current ball position
    cv2.circle(frame, (int(recent[-1][0]), int(recent[-1][1])), 10,
               COLOR_BALL_TRAJECTORY, 2)
    return frame


def draw_ball_state(frame: np.ndarray, ball_state) -> np.ndarray:
    """Draw ball detection label and bounding box."""
    if ball_state is None or not ball_state.detected:
        return frame
    color = COLOR_BALL_TRAJECTORY if ball_state.state == "IN_FLIGHT" else (0, 200, 255)
    cv2.putText(frame, f"Ball: {ball_state.state}", (5, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if ball_state.bbox:
        x1, y1, x2, y2 = [int(v) for v in ball_state.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


REVIEW_THUMB_W = 480  # width each thumbnail is scaled to in the review panel


def build_shot_review_panel(loading_frame: np.ndarray, release_frame: np.ndarray,
                             load_angles: dict, release_angles: dict) -> np.ndarray:
    """Build a side-by-side shot review panel with labelled thumbnails and angle strips."""

    def scale_to_width(img, target_w):
        h, w = img.shape[:2]
        scale = target_w / w
        return cv2.resize(img, (target_w, int(h * scale)))

    left = scale_to_width(loading_frame, REVIEW_THUMB_W)
    right = scale_to_width(release_frame, REVIEW_THUMB_W)

    # Match heights
    th = max(left.shape[0], right.shape[0])
    def pad_height(img, target_h):
        h, w = img.shape[:2]
        if h < target_h:
            pad = np.zeros((target_h - h, w, 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img
    left = pad_height(left, th)
    right = pad_height(right, th)

    divider = np.zeros((th, 4, 3), dtype=np.uint8)
    panel_w = REVIEW_THUMB_W * 2 + 4

    # --- Header bar ---
    header_h = 40
    header = np.full((header_h, panel_w, 3), 40, dtype=np.uint8)
    header_text = "SHOT REVIEW  |  Click a frame to expand"
    cv2.putText(header, header_text, (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    # --- Label bars ---
    label_h = 32
    left_label = np.full((label_h, REVIEW_THUMB_W, 3), (20, 130, 180), dtype=np.uint8)  # amber-ish
    right_label = np.full((label_h, REVIEW_THUMB_W, 3), (30, 140, 30), dtype=np.uint8)  # green-ish
    cv2.putText(left_label, "SHOOTING POSITION", (8, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(right_label, "RELEASED", (8, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    label_row = np.hstack([left_label, np.zeros((label_h, 4, 3), dtype=np.uint8), right_label])

    # --- Thumbnail row ---
    thumb_row = np.hstack([left, divider, right])

    # --- Angle strip ---
    strip_h = 56
    strip = np.full((strip_h, panel_w, 3), 25, dtype=np.uint8)
    le = f"{load_angles.get('elbow'):.1f}" + chr(176) if load_angles.get('elbow') is not None else "N/A"
    lk = f"{load_angles.get('knee'):.1f}" + chr(176) if load_angles.get('knee') is not None else "N/A"
    re_ = f"{release_angles.get('elbow'):.1f}" + chr(176) if release_angles.get('elbow') is not None else "N/A"
    rk = f"{release_angles.get('knee'):.1f}" + chr(176) if release_angles.get('knee') is not None else "N/A"
    cv2.putText(strip, f"Elbow: {le}   Knee: {lk}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(strip, f"Elbow: {re_}   Knee: {rk}", (REVIEW_THUMB_W + 14, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(strip, "Press any key in expanded view to close it", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    return np.vstack([header, label_row, thumb_row, strip])


def draw_overlay_text(frame: np.ndarray, text: str, color_bgr: tuple) -> np.ndarray:
    """Draw large centered overlay text with a semi-transparent dark background."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 2.0
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h // 2 + th // 2

    # Semi-transparent background rectangle
    pad = 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, text, (x, y), font, scale, color_bgr, thickness, cv2.LINE_AA)
    return frame


def compose_frame(frame: np.ndarray, analysis: FrameAnalysis,
                  trajectory: list, ball_trajectory: list = None,
                  overlay_text: str = None, overlay_color: tuple = (255, 255, 255)) -> np.ndarray:
    out = frame.copy()

    # Draw skeleton for primary person
    if analysis.persons:
        out = draw_skeleton(out, analysis.persons[0])

    out = draw_angles(out, analysis)
    out = draw_wrist_trajectory(out, trajectory)
    if ball_trajectory:
        out = draw_ball_trajectory(out, ball_trajectory)
    out = draw_ball_state(out, analysis.ball_state)
    out = draw_shot_phase(out, analysis.shot_phase, analysis.frame_index, analysis.timestamp)

    # Hand landmarks hidden — YOLO pose skeleton already covers the arm

    # Draw held object nodes
    for obj in (analysis.held_objects or []):
        out = draw_held_object_nodes(out, obj)

    if overlay_text:
        out = draw_overlay_text(out, overlay_text, overlay_color)

    return out
