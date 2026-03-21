import math
import numpy as np
from typing import Optional


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees.
    Returns nan for missing/invalid points."""
    if a is None or b is None or c is None:
        return float('nan')
    ax, ay = a
    bx, by = b
    cx, cy = c
    # Vectors from b
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return float('nan')
    cos_val = dot / (mag1 * mag2)
    cos_val = max(-1.0, min(1.0, cos_val))  # clamp to avoid NaN from arccos
    return math.degrees(math.acos(cos_val))


def calculate_vector_angle(vector: tuple, reference: tuple = (1.0, 0.0)) -> float:
    """Angle of vector relative to reference vector, in degrees.
    Caller should invert Y for image→math coordinate conversion."""
    vx, vy = vector
    rx, ry = reference
    mag_v = math.sqrt(vx ** 2 + vy ** 2)
    mag_r = math.sqrt(rx ** 2 + ry ** 2)
    if mag_v < 1e-6 or mag_r < 1e-6:
        return float('nan')
    dot = vx * rx + vy * ry
    cos_val = max(-1.0, min(1.0, dot / (mag_v * mag_r)))
    angle = math.degrees(math.acos(cos_val))
    # Use cross product sign to get signed angle
    cross = vx * ry - vy * rx
    if cross < 0:
        angle = -angle
    return angle


def smooth_trajectory(positions: list, window: int = 5) -> list:
    """Moving average smoothing of (x, y) position list."""
    if len(positions) < 2:
        return positions
    smoothed = []
    half = window // 2
    for i in range(len(positions)):
        start = max(0, i - half)
        end = min(len(positions), i + half + 1)
        chunk = [p for p in positions[start:end] if p is not None]
        if chunk:
            avg_x = sum(p[0] for p in chunk) / len(chunk)
            avg_y = sum(p[1] for p in chunk) / len(chunk)
            smoothed.append((avg_x, avg_y))
        else:
            smoothed.append(positions[i])
    return smoothed


def compute_velocity(positions: list, fps: float) -> list:
    """Per-frame velocity via central differences. Returns list of (vx, vy) tuples."""
    if len(positions) < 2:
        return [(0.0, 0.0)] * len(positions)
    velocities = []
    n = len(positions)
    for i in range(n):
        if i == 0:
            dx = positions[1][0] - positions[0][0]
            dy = positions[1][1] - positions[0][1]
        elif i == n - 1:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            dx = (positions[i + 1][0] - positions[i - 1][0]) / 2.0
            dy = (positions[i + 1][1] - positions[i - 1][1]) / 2.0
        velocities.append((dx * fps, dy * fps))
    return velocities
