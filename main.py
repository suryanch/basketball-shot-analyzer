#!/usr/bin/env python3
"""Basketball Shot Analyzer — CLI entry point."""
import argparse
import os
import math
import sys
import cv2
import numpy as np

import config
from pose_analyzer import PoseAnalyzer
from shot_detector import ShotDetector
from visualizer import compose_frame, build_shot_review_panel
from reporter import Reporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Basketball Shot Analyzer using YOLO pose estimation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", metavar="PATH", help="Video file input")
    group.add_argument("--webcam", metavar="INDEX", nargs="?", const=0, type=int,
                       help="Webcam input (default index 0)")
    group.add_argument("--image", metavar="PATH", help="Single image input")

    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        help=f"YOLO pose model (default: {config.DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=config.MIN_CONFIDENCE,
                        help=f"Detection confidence threshold (default: {config.MIN_CONFIDENCE})")
    parser.add_argument("--iou", type=float, default=config.DEFAULT_IOU,
                        help=f"NMS IoU threshold (default: {config.DEFAULT_IOU})")
    parser.add_argument("--output", metavar="PATH",
                        help="Output video/image path (auto-named if omitted)")
    parser.add_argument("--report", default=config.DEFAULT_REPORT_FILE,
                        help=f"Report JSON path (default: {config.DEFAULT_REPORT_FILE})")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress real-time display window")
    parser.add_argument("--static-analysis", action="store_true",
                        help="Skip state machine; report angles from a single frame directly")
    parser.add_argument("--fps", type=float, default=config.DEFAULT_FPS,
                        help=f"Override FPS for webcam (default: {config.DEFAULT_FPS})")
    parser.add_argument("--skip", type=int, default=config.INFERENCE_SKIP_FRAMES,
                        help=f"Run ML inference every Nth frame (default: {config.INFERENCE_SKIP_FRAMES})")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated output video (disabled by default for performance)")
    return parser.parse_args()


def resolve_input_source(args):
    """Returns (source_label, cap_or_path, fps, is_image)."""
    if args.image:
        return args.image, args.image, 1.0, True
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        return args.video, cap, fps, False
    # webcam
    idx = args.webcam if args.webcam is not None else 0
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open webcam index {idx}")
    return f"webcam:{idx}", cap, args.fps, False


def make_output_path(args, source_label: str, is_image: bool) -> str:
    if args.output:
        return args.output
    base = os.path.splitext(os.path.basename(source_label))[0]
    if "webcam" in source_label:
        base = "webcam"
    ext = ".jpg" if is_image else ".mp4"
    return f"{base}_analyzed{ext}"


def frame_generator(cap, is_image: bool, source_label: str, is_webcam: bool = False):
    """Yields (frame_idx, frame) tuples."""
    if is_image:
        frame = cv2.imread(source_label)
        if frame is None:
            sys.exit(f"[ERROR] Cannot read image: {source_label}")
        yield 0, frame
        return
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1


def create_writer(output_path: str, cap, fps: float, is_image: bool):
    if is_image:
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def show_shot_review(loading_frame, release_frame, load_angles, release_angles):
    """Open a non-blocking Shot Review window with clickable thumbnails."""
    panel = build_shot_review_panel(loading_frame, release_frame, load_angles, release_angles)
    win = "Shot Review"
    cv2.imshow(win, panel)
    panel_w = panel.shape[1]

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < panel_w // 2:
            cv2.imshow("Loading Frame (Shooting Position)", loading_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("Loading Frame (Shooting Position)")
        else:
            cv2.imshow("Release Frame", release_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("Release Frame")

    cv2.setMouseCallback(win, on_click)


def run():
    args = parse_args()
    source_label, source, fps, is_image = resolve_input_source(args)
    output_path = make_output_path(args, source_label, is_image)

    analyzer = PoseAnalyzer(model_path=args.model, conf=args.conf, iou=args.iou, hand_mode="yolo")
    detector = ShotDetector()
    reporter = Reporter()

    is_webcam = args.webcam is not None
    writer = None
    if not is_image:
        writer = create_writer(output_path, source, fps, is_image)

    wrist_trajectory: list = []
    ball_trajectory: list = []
    total_frames = 0
    last_analysis = None
    release_capture_count = 0
    loading_capture_count = 0
    from collections import deque
    frame_buffer: deque = deque(maxlen=config.FRAME_BUFFER_SIZE)
    overlay_text = None
    overlay_color = (255, 255, 255)
    prev_phase = "IDLE"
    prev_elbow_angle = None        # smoothed elbow angle from previous frame (shooting side)
    shooting_range_counter = 0     # consecutive frames with elbow below entry threshold (debounce)
    SHOOTING_DEBOUNCE = 6          # require ~0.2s of stable elbow angle before triggering
    SHOOT_ENTER_THRESHOLD = 90.0   # elbow below this triggers "Shooting Position"
    RELEASE_THRESHOLD = 110.0      # elbow above this triggers "Released"
    in_shooting_mode = False       # hysteresis state
    OVERLAY_EXIT_FRAMES = 10       # frames out of range before text disappears
    overlay_exit_counter = 0       # counts frames since angle left the active range
    elbow_angle_buffer: list = []  # rolling buffer for smoothing elbow angle
    ELBOW_SMOOTH_WINDOW = 5        # smooth over last 5 frames
    shooting_screenshot_count = 0
    released_screenshot_count = 0
    shooting_capture_remaining = 0   # frames left to capture for shooting position
    released_capture_remaining = 0   # frames left to capture for released
    SCREENSHOT_BURST = 5             # number of frames to capture per trigger
    last_loading_annotated = None
    last_release_annotated = None
    last_load_angles = {}
    last_release_angles = {}

    print(f"[INFO] Analyzing: {source_label}")
    if args.save_video:
        print(f"[INFO] Output   : {output_path}")
    print(f"[INFO] Inference every {args.skip} frame(s)  |  Press 'q' to quit\n")

    for frame_idx, frame in frame_generator(source, is_image, source_label, is_webcam):
        # Skip ML inference on non-processed frames; reuse last result
        if last_analysis is not None and frame_idx % args.skip != 0:
            annotated = compose_frame(frame, last_analysis, wrist_trajectory, ball_trajectory,
                                      overlay_text=overlay_text,
                                      overlay_color=overlay_color)
            if not args.no_display:
                cv2.imshow("Basketball Shot Analyzer", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer and args.save_video:
                writer.write(annotated)
            total_frames += 1
            continue

        analysis = analyzer.analyze_frame(frame, frame_idx, fps)

        # Determine which knee/elbow angles to pass to detector (shooting side)
        if analysis.shooting_side == "right":
            knee_angle = analysis.knee_angle_right
            elbow_angle = analysis.elbow_angle_right
        else:
            knee_angle = analysis.knee_angle_left
            elbow_angle = analysis.elbow_angle_left

        phase, release_arc, load_event = detector.update(
            frame_idx, analysis.timestamp,
            analysis.wrist_pos, knee_angle, elbow_angle,
            ball_state=analysis.ball_state,
        )
        analysis.shot_phase = phase
        if release_arc is not None:
            analysis.release_arc = release_arc

        # Smooth elbow angle over last N frames to reduce noise
        raw_elbow_valid = elbow_angle is not None and not (isinstance(elbow_angle, float) and math.isnan(elbow_angle))
        if raw_elbow_valid:
            elbow_angle_buffer.append(elbow_angle)
            if len(elbow_angle_buffer) > ELBOW_SMOOTH_WINDOW:
                elbow_angle_buffer.pop(0)
            elbow_angle = sum(elbow_angle_buffer) / len(elbow_angle_buffer)

        # Elbow-angle-based overlay triggers (primary filter)
        # Valid elbow angle: not None and not NaN
        elbow_valid = elbow_angle is not None and not (isinstance(elbow_angle, float) and math.isnan(elbow_angle))

        if elbow_valid:
            # Clear overlay if angle has been out of the active range for too long
            if overlay_text == "Shooting Position":
                if elbow_angle >= SHOOT_ENTER_THRESHOLD:
                    overlay_exit_counter += 1
                    if overlay_exit_counter >= OVERLAY_EXIT_FRAMES:
                        overlay_text = None
                        in_shooting_mode = False
                        overlay_exit_counter = 0
                else:
                    overlay_exit_counter = 0
            elif overlay_text == "Released":
                if elbow_angle <= RELEASE_THRESHOLD:
                    overlay_exit_counter += 1
                    if overlay_exit_counter >= OVERLAY_EXIT_FRAMES:
                        overlay_text = None
                        overlay_exit_counter = 0
                else:
                    overlay_exit_counter = 0

            # Trigger new overlays
            if not in_shooting_mode:
                if elbow_angle < SHOOT_ENTER_THRESHOLD:
                    shooting_range_counter += 1
                    if shooting_range_counter >= SHOOTING_DEBOUNCE:
                        in_shooting_mode = True
                        shooting_range_counter = 0
                        overlay_text = "Shooting Position"
                        overlay_color = (0, 215, 255)   # amber/gold
                        overlay_exit_counter = 0
                        shooting_capture_remaining = SCREENSHOT_BURST
                else:
                    shooting_range_counter = 0
            else:
                if elbow_angle > RELEASE_THRESHOLD:
                    in_shooting_mode = False
                    overlay_text = "Released"
                    overlay_color = (0, 255, 100)   # green
                    overlay_exit_counter = 0
                    released_capture_remaining = SCREENSHOT_BURST

        prev_elbow_angle = elbow_angle if elbow_valid else prev_elbow_angle
        prev_phase = phase

        # Update wrist trajectory
        if analysis.wrist_pos:
            wrist_trajectory.append(analysis.wrist_pos)
        else:
            wrist_trajectory.append(None)
        if len(wrist_trajectory) > 50:
            wrist_trajectory = wrist_trajectory[-50:]

        if analysis.ball_state and analysis.ball_state.center:
            ball_trajectory.append(analysis.ball_state.center)
        else:
            ball_trajectory.append(None)
        if len(ball_trajectory) > 50:
            ball_trajectory = ball_trajectory[-50:]

        annotated = compose_frame(frame, analysis, wrist_trajectory, ball_trajectory,
                                  overlay_text=overlay_text, overlay_color=overlay_color)
        frame_buffer.append((frame_idx, annotated.copy()))

        if shooting_capture_remaining > 0:
            shooting_screenshot_count += 1
            path = os.path.expanduser(f"~/Desktop/shooting_position_{shooting_screenshot_count:03d}.jpg")
            cv2.imwrite(path, annotated)
            print(f"[SHOT POS] Frame {frame_idx} — {path}")
            shooting_capture_remaining -= 1

        if released_capture_remaining > 0:
            released_screenshot_count += 1
            path = os.path.expanduser(f"~/Desktop/released_{released_screenshot_count:03d}.jpg")
            cv2.imwrite(path, annotated)
            print(f"[RELEASED] Frame {frame_idx} — {path}")
            released_capture_remaining -= 1

        # Save loading frame (max knee bend / pre-jump position) when COCKING→RELEASE fires
        if load_event is not None:
            loading_capture_count += 1
            loading_path = f"loading_{loading_capture_count:03d}.jpg"
            target_idx = load_event["frame_idx"]
            saved = False
            for buf_idx, buf_frame in frame_buffer:
                if buf_idx == target_idx:
                    last_loading_annotated = buf_frame.copy()
                    cv2.imwrite(loading_path, buf_frame)
                    saved = True
                    break
            last_load_angles = {"elbow": load_event["elbow"], "knee": load_event["knee"]}
            elbow_str = f"{load_event['elbow']:.1f}°" if load_event['elbow'] is not None else "N/A"
            knee_str = f"{load_event['knee']:.1f}°" if load_event['knee'] is not None else "N/A"
            print(f"[LOAD]    Frame {target_idx} — {loading_path if saved else '(frame not in buffer)'}")
            print(f"          Elbow: {elbow_str}  |  Knee: {knee_str}  (max knee bend / pre-jump position)")

        # Save release frame when ball becomes IN_FLIGHT
        if (analysis.ball_state and analysis.ball_state.just_released):
            release_capture_count += 1
            release_path = f"release_{release_capture_count:03d}.jpg"
            cv2.imwrite(release_path, annotated)
            if analysis.shooting_side == "right":
                rel_elbow = analysis.elbow_angle_right
                rel_knee = analysis.knee_angle_right
            else:
                rel_elbow = analysis.elbow_angle_left
                rel_knee = analysis.knee_angle_left
            last_release_annotated = annotated.copy()
            last_release_angles = {"elbow": rel_elbow, "knee": rel_knee}
            elbow_str = f"{rel_elbow:.1f}°" if rel_elbow is not None else "N/A"
            knee_str = f"{rel_knee:.1f}°" if rel_knee is not None else "N/A"
            print(f"[RELEASE] Frame {frame_idx} — {release_path}")
            print(f"          Elbow: {elbow_str}  |  Knee: {knee_str}  |  Side: {analysis.shooting_side}")
            if last_loading_annotated is not None and not args.no_display:
                show_shot_review(last_loading_annotated, last_release_annotated,
                                 last_load_angles, last_release_angles)

        last_analysis = analysis
        total_frames += 1

        if not args.no_display:
            cv2.imshow("Basketball Shot Analyzer", annotated)
            key = cv2.waitKey(1 if not is_image else 0) & 0xFF
            if key == ord('q'):
                break

        if is_image:
            cv2.imwrite(output_path, annotated)
            print(f"[INFO] Annotated image saved: {output_path}")
        elif writer and args.save_video:
            writer.write(annotated)

    # Cleanup
    if writer:
        writer.release()
    if not is_image and hasattr(source, "release"):
        source.release()
    cv2.destroyAllWindows()

    if not is_image and args.save_video:
        print(f"[INFO] Annotated video saved: {output_path}")

    # Generate report
    if args.static_analysis and last_analysis is not None:
        report = reporter.compile_static_report(last_analysis, source_label)
        reporter.save_report(report, args.report)
        reporter.print_static_summary(report)
    else:
        report = reporter.compile_report(
            detector.shot_events, source_label, total_frames, fps
        )
        reporter.save_report(report, args.report)
        reporter.print_summary(report)


if __name__ == "__main__":
    run()
