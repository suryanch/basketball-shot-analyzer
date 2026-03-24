import json
import math
import statistics
from typing import Optional

import config
from shot_detector import ShotEvent


def rule_elbow_angle(elbow_angle: Optional[float]) -> Optional[str]:
    if elbow_angle is None or math.isnan(elbow_angle):
        return None
    diff = elbow_angle - config.IDEAL_ELBOW_ANGLE
    if diff > config.ELBOW_ANGLE_TOLERANCE:
        return (f"Elbow too extended at release ({elbow_angle:.0f}°, ideal ~{config.IDEAL_ELBOW_ANGLE:.0f}°). "
                "Tuck the elbow under the ball for better control.")
    if diff < -config.ELBOW_ANGLE_TOLERANCE:
        return (f"Elbow too tucked ({elbow_angle:.0f}°, ideal ~{config.IDEAL_ELBOW_ANGLE:.0f}°). "
                "Allow a more natural elbow angle to generate arc.")
    return None


def rule_knee_bend(knee_angle: Optional[float]) -> Optional[str]:
    if knee_angle is None or math.isnan(knee_angle):
        return None
    if knee_angle < config.IDEAL_KNEE_MIN:
        return (f"Excessive knee bend ({knee_angle:.0f}°, ideal {config.IDEAL_KNEE_MIN:.0f}–{config.IDEAL_KNEE_MAX:.0f}°). "
                "Reduce squat depth to avoid losing upward momentum.")
    if knee_angle > config.IDEAL_KNEE_MAX:
        return (f"Insufficient knee bend ({knee_angle:.0f}°, ideal {config.IDEAL_KNEE_MIN:.0f}–{config.IDEAL_KNEE_MAX:.0f}°). "
                "Bend knees more to load power into the shot.")
    return None


def rule_release_arc(release_arc: Optional[float]) -> Optional[str]:
    if release_arc is None or math.isnan(release_arc):
        return None
    if release_arc < config.IDEAL_RELEASE_ARC_MIN:
        return (f"Release arc too flat ({release_arc:.1f}°, ideal {config.IDEAL_RELEASE_ARC_MIN:.0f}–{config.IDEAL_RELEASE_ARC_MAX:.0f}°). "
                "Aim higher to increase the entry angle into the basket.")
    if release_arc > config.IDEAL_RELEASE_ARC_MAX:
        return (f"Release arc too steep ({release_arc:.1f}°, ideal {config.IDEAL_RELEASE_ARC_MIN:.0f}–{config.IDEAL_RELEASE_ARC_MAX:.0f}°). "
                "Reduce arc angle slightly for a more consistent trajectory.")
    return None


def rule_wrist_snap(trajectory: list) -> Optional[str]:
    """Heuristic: check deceleration in last portion of trajectory for wrist snap."""
    valid = [p for p in trajectory if p is not None]
    if len(valid) < 6:
        return None
    # Compute speeds in first half vs last third
    mid = len(valid) // 2
    def avg_speed(pts):
        speeds = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            speeds.append(math.sqrt(dx**2 + dy**2))
        return sum(speeds) / len(speeds) if speeds else 0
    early_speed = avg_speed(valid[:mid])
    late_speed = avg_speed(valid[mid:])
    if early_speed > 0 and late_speed / early_speed > 0.8:
        return ("Wrist snap may be insufficient. The wrist should decelerate sharply after release, "
                "indicating a strong follow-through snap.")
    return None


def rule_trajectory_consistency(arcs: list) -> Optional[str]:
    valid = [a for a in arcs if a is not None and not math.isnan(a)]
    if len(valid) < 2:
        return None
    std = statistics.stdev(valid)
    if std > config.TRAJECTORY_CONSISTENCY_STD:
        return (f"Inconsistent release arc across shots (std dev {std:.1f}°). "
                "Focus on a repeatable, consistent release point.")
    return None


class Reporter:
    def compile_static_report(self, analysis, source: str) -> dict:
        """Generate a biomechanics report from a single static frame's angles."""
        shooting_side = analysis.shooting_side
        if shooting_side == "right":
            elbow = analysis.elbow_angle_right
            knee = analysis.knee_angle_right
        else:
            elbow = analysis.elbow_angle_left
            knee = analysis.knee_angle_left

        suggestions = []
        for s in [rule_elbow_angle(elbow), rule_knee_bend(knee)]:
            if s:
                suggestions.append(s)

        metrics = {
            "shooting_side": shooting_side,
            "elbow_angle_deg": round(elbow, 1) if elbow and not __import__('math').isnan(elbow) else None,
            "knee_angle_deg": round(knee, 1) if knee and not __import__('math').isnan(knee) else None,
            "elbow_angle_left_deg": round(analysis.elbow_angle_left, 1) if analysis.elbow_angle_left and not __import__('math').isnan(analysis.elbow_angle_left) else None,
            "elbow_angle_right_deg": round(analysis.elbow_angle_right, 1) if analysis.elbow_angle_right and not __import__('math').isnan(analysis.elbow_angle_right) else None,
            "knee_angle_left_deg": round(analysis.knee_angle_left, 1) if analysis.knee_angle_left and not __import__('math').isnan(analysis.knee_angle_left) else None,
            "knee_angle_right_deg": round(analysis.knee_angle_right, 1) if analysis.knee_angle_right and not __import__('math').isnan(analysis.knee_angle_right) else None,
            "wrist_position": analysis.wrist_pos,
        }

        note = ("Release arc cannot be computed from a single frame. "
                "Use --video for full shot trajectory analysis.")

        return {
            "mode": "static_analysis",
            "metadata": {"source": source, "model": config.DEFAULT_MODEL},
            "metrics": metrics,
            "suggestions": suggestions,
            "grade": self.grade_shot(suggestions),
            "notes": [note],
        }

    def print_static_summary(self, report: dict):
        m = report["metrics"]
        print("\n" + "=" * 60)
        print("  STATIC SHOT ANALYSIS REPORT")
        print("=" * 60)
        print(f"  Source        : {report['metadata']['source']}")
        print(f"  Shooting side : {m['shooting_side']}")
        print(f"  Elbow angle   : {m['elbow_angle_deg']}°  (ideal ~{config.IDEAL_ELBOW_ANGLE:.0f}°)")
        print(f"  Knee angle    : {m['knee_angle_deg']}°  (ideal {config.IDEAL_KNEE_MIN:.0f}–{config.IDEAL_KNEE_MAX:.0f}°)")
        print(f"  Grade         : {report['grade']}")
        if report["suggestions"]:
            print("\n  Suggestions:")
            for i, s in enumerate(report["suggestions"], 1):
                print(f"    {i}. {s}")
        else:
            print("\n  No issues detected — angles look good!")
        for note in report.get("notes", []):
            print(f"\n  Note: {note}")
        print("=" * 60 + "\n")


    def generate_suggestions(self, event: ShotEvent) -> list:
        suggestions = []
        for rule in [
            rule_elbow_angle(event.elbow_angle_at_set),
            rule_knee_bend(event.knee_angle_at_initiation),
            rule_release_arc(event.effective_release_arc),
            rule_wrist_snap(event.trajectory),
        ]:
            if rule:
                suggestions.append(rule)
        return suggestions

    def grade_shot(self, suggestions: list) -> str:
        n = len(suggestions)
        if n == 0:
            return "good"
        if n <= 2:
            return "needs_work"
        return "poor"

    def compile_report(self, events: list, source: str,
                       total_frames: int, fps: float) -> dict:
        shots = []
        all_arcs = []
        all_suggestions = []

        for event in events:
            sugg = self.generate_suggestions(event)
            grade = self.grade_shot(sugg)
            all_suggestions.extend(sugg)
            if event.effective_release_arc:
                all_arcs.append(event.effective_release_arc)

            shots.append({
                "frame": event.frame,
                "timestamp_s": round(event.timestamp, 3),
                "elbow_angle_at_load_deg": round(event.elbow_angle_at_load, 1) if event.elbow_angle_at_load else None,
                "knee_angle_at_load_deg": round(event.knee_angle_at_load, 1) if event.knee_angle_at_load else None,
                "load_frame_index": event.load_frame_index,
                "elbow_angle_at_set_deg": round(event.elbow_angle_at_set, 1) if event.elbow_angle_at_set else None,
                "knee_angle_at_initiation_deg": round(event.knee_angle_at_initiation, 1) if event.knee_angle_at_initiation else None,
                "elbow_angle_at_release_deg": round(event.elbow_angle_at_release, 1) if event.elbow_angle_at_release else None,
                "knee_angle_at_release_deg": round(event.knee_angle_at_release, 1) if event.knee_angle_at_release else None,
                "release_frame_index": event.release_frame_index,
                "release_arc_deg": round(event.effective_release_arc, 1) if event.effective_release_arc else None,
                "release_arc_source": "ball" if event.ball_release_arc is not None else "wrist",
                "suggestions": sugg,
                "grade": grade,
            })

        # Most common issue
        from collections import Counter
        most_common_issue = None
        if all_suggestions:
            # Extract the first sentence of each suggestion as a category
            categories = [s.split(".")[0] for s in all_suggestions]
            most_common_issue = Counter(categories).most_common(1)[0][0]

        # Consistency check
        consistency_note = rule_trajectory_consistency(all_arcs)

        summary = {
            "total_shots_detected": len(events),
            "average_release_arc_deg": round(sum(all_arcs) / len(all_arcs), 1) if all_arcs else None,
            "most_common_issue": most_common_issue,
            "consistency_note": consistency_note,
            "grade_distribution": {
                "good": sum(1 for s in shots if s["grade"] == "good"),
                "needs_work": sum(1 for s in shots if s["grade"] == "needs_work"),
                "poor": sum(1 for s in shots if s["grade"] == "poor"),
            },
        }

        return {
            "metadata": {
                "source": source,
                "total_frames": total_frames,
                "fps": fps,
                "duration_s": round(total_frames / fps, 2) if fps > 0 else None,
                "model": config.DEFAULT_MODEL,
            },
            "shots": shots,
            "summary": summary,
        }

    def save_report(self, report: dict, path: str = config.DEFAULT_REPORT_FILE):
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {path}")

    def print_summary(self, report: dict):
        meta = report["metadata"]
        summary = report["summary"]
        shots = report["shots"]

        print("\n" + "=" * 60)
        print("  BASKETBALL SHOT ANALYSIS REPORT")
        print("=" * 60)
        print(f"  Source      : {meta['source']}")
        print(f"  Duration    : {meta.get('duration_s', 'N/A')}s  ({meta['total_frames']} frames @ {meta['fps']} fps)")
        print(f"  Shots found : {summary['total_shots_detected']}")
        if summary["average_release_arc_deg"] is not None:
            print(f"  Avg arc     : {summary['average_release_arc_deg']}°  (ideal {config.IDEAL_RELEASE_ARC_MIN}–{config.IDEAL_RELEASE_ARC_MAX}°)")

        grades = summary["grade_distribution"]
        print(f"  Grades      : {grades['good']} good / {grades['needs_work']} needs_work / {grades['poor']} poor")

        if summary["most_common_issue"]:
            print(f"\n  Top issue   : {summary['most_common_issue']}")
        if summary["consistency_note"]:
            print(f"  Consistency : {summary['consistency_note']}")

        if shots:
            print("\n  Per-shot breakdown:")
            print(f"  {'#':<4} {'Time':>6}  {'E@load':>7}  {'K@load':>7}  {'E@set':>6}  {'K@set':>6}  {'E@rel':>6}  {'K@rel':>6}  {'Arc':>6}  {'Grade':<12}")
            print("  " + "-" * 85)
            for i, s in enumerate(shots, 1):
                e_load = f"{s['elbow_angle_at_load_deg']}°" if s['elbow_angle_at_load_deg'] else "N/A"
                k_load = f"{s['knee_angle_at_load_deg']}°" if s['knee_angle_at_load_deg'] else "N/A"
                elbow_set = f"{s['elbow_angle_at_set_deg']}°" if s['elbow_angle_at_set_deg'] else "N/A"
                knee_set = f"{s['knee_angle_at_initiation_deg']}°" if s['knee_angle_at_initiation_deg'] else "N/A"
                elbow_rel = f"{s['elbow_angle_at_release_deg']}°" if s['elbow_angle_at_release_deg'] else "N/A"
                knee_rel = f"{s['knee_angle_at_release_deg']}°" if s['knee_angle_at_release_deg'] else "N/A"
                arc = f"{s['release_arc_deg']}°" if s['release_arc_deg'] else "N/A"
                print(f"  {i:<4} {s['timestamp_s']:>6.2f}s  {e_load:>7}  {k_load:>7}  {elbow_set:>6}  {knee_set:>6}  {elbow_rel:>6}  {knee_rel:>6}  {arc:>6}  {s['grade']:<12}")

        print("=" * 60 + "\n")
