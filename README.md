# Basketball Shot Analyzer

Real-time basketball shooting mechanics analyzer using YOLO11 pose estimation, MediaPipe hand tracking, and ball trajectory analysis. Detects joint angles, tracks the ball through release, measures release arc, and generates improvement suggestions based on biomechanics benchmarks.

## Features

- **Pose estimation** — COCO 17-keypoint skeleton with elbow and knee angle overlays
- **Hand tracking** — MediaPipe 21-point hand landmarks always visible
- **Held object detection** — YOLO-based detection of objects in hand with fitted node overlays
- **Ball tracking** — Full-frame ball detection with HELD / IN_FLIGHT state machine
- **Release arc measurement** — Arc angle computed from the ball's actual flight trajectory
- **Release frame capture** — Annotated frame saved as `release_NNN.jpg` at the moment of release
- **Shot state machine** — IDLE → COCKING → RELEASE → FOLLOW_THROUGH phase detection
- **Improvement report** — JSON report with per-shot grades and biomechanics suggestions
- **Three input modes** — Webcam (real-time), video file, or single image

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| pip | latest recommended |
| Webcam | required for live mode only |
| OS | macOS, Linux, or Windows |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/suryanch/basketball-shot-analyzer.git
cd basketball-shot-analyzer
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Model downloads (automatic)

The following models are downloaded automatically on first run — no manual steps needed:

| Model | Size | Purpose |
|---|---|---|
| `yolo11n-pose.pt` | ~6 MB | Body pose estimation (17 keypoints) |
| `yolo11n.pt` | ~5.5 MB | Object + ball detection |
| `hand_landmarker.task` | ~7.5 MB | MediaPipe hand landmarks |

Models are cached in the project directory and reused on subsequent runs.

---

## Usage

### Webcam (real-time)

```bash
python main.py --webcam
```

Press **`q`** to stop. The analyzed video is saved as `webcam_analyzed.mp4`.

### Video file

```bash
python main.py --video path/to/clip.mp4
```

Output saved as `clip_analyzed.mp4` alongside the input, unless `--output` is specified.

### Single image

```bash
python main.py --image path/to/shot.jpg
```

Output saved as `shot_analyzed.jpg`. Use `--static-analysis` to skip the shot state machine and report angles directly from the single frame.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--webcam [INDEX]` | `0` | Webcam input (index optional, default 0) |
| `--video PATH` | — | Video file input |
| `--image PATH` | — | Single image input |
| `--model MODEL` | `yolo11n-pose.pt` | YOLO pose model path or name |
| `--conf FLOAT` | `0.5` | Keypoint detection confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--output PATH` | auto-named | Output video or image path |
| `--report PATH` | `shot_report.json` | JSON report output path |
| `--fps FLOAT` | `30.0` | Override FPS (webcam only) |
| `--no-display` | off | Suppress the real-time preview window |
| `--static-analysis` | off | Report angles from a single frame (image mode) |

---

## Output Files

| File | Description |
|---|---|
| `*_analyzed.mp4 / .jpg` | Annotated video or image with skeleton, angles, ball trajectory |
| `shot_report.json` | JSON report with per-shot metrics, grades, and suggestions |
| `release_001.jpg`, `release_002.jpg`, … | Annotated frame captured at the exact moment of ball release |

---

## Biomechanics Benchmarks

The suggestion engine flags shots that fall outside these ranges:

| Metric | Ideal Range |
|---|---|
| Elbow angle at set point | ~90° |
| Knee bend at initiation | 90° – 120° |
| Release arc | 45° – 52° above horizontal |

Shot grades: **good** (0 flags) / **needs_work** (1–2 flags) / **poor** (3+ flags)

---

## Project Structure

```
basketball_shot_analyzer/
├── main.py               # CLI entry point and frame loop
├── config.py             # All shared constants and thresholds
├── pose_analyzer.py      # YOLO pose inference + keypoint extraction
├── angle_calculator.py   # Joint angle math (dot product / arccos)
├── shot_detector.py      # Shot state machine + ShotEvent dataclass
├── ball_tracker.py       # Full-frame ball detection + HELD/IN_FLIGHT logic
├── hand_detector.py      # MediaPipe hand landmarks + YOLO object detection
├── visualizer.py         # OpenCV + Pillow overlay drawing
├── reporter.py           # Suggestion rules, grading, JSON report
├── requirements.txt
└── .gitignore
```

---

## Configuration

All thresholds and model settings are in `config.py`. Key values:

```python
MIN_CONFIDENCE = 0.5          # Keypoint confidence cutoff
BALL_CONF_THRESHOLD = 0.25    # Ball detection confidence
BALL_HELD_DISTANCE_PX = 120   # Distance (px) to classify ball as HELD
BALL_MIN_SIZE_PX = 40         # Minimum ball bbox size to reject false positives
IDEAL_ELBOW_ANGLE = 90.0      # Degrees
IDEAL_RELEASE_ARC_MIN = 45.0  # Degrees above horizontal
IDEAL_RELEASE_ARC_MAX = 52.0  # Degrees above horizontal
```

---

## Troubleshooting

**Webcam not opening**
```bash
python main.py --webcam 1   # try index 1 or 2 if 0 fails
```

**`cv2` or `mediapipe` not found**
```bash
pip install -r requirements.txt
```

**Degree symbol shows as `??`**
Ensure Pillow is installed: `pip install Pillow`

**Ball not detected / too many false positives**
Adjust `BALL_CONF_THRESHOLD` and `BALL_MIN_SIZE_PX` in `config.py`.

**Slow on CPU**
Use a smaller model: `--model yolo11n-pose.pt` (default) is already the nano variant. For faster inference, reduce input resolution or run on a GPU-enabled machine.
