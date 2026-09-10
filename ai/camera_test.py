"""
camera_test.py
--------------
PlaySafe AI - Advanced Kinematic Movement & Fall Screening MVP.

Features:
- CYBER-HUD: Holographic bloom, grid overlays, dynamic scanlines, and progress bars.
- DAILY LIFE & WEIGHTLIFTING: Text Neck, Locked Knees, Spinal Torsion, etc.
- WIKIPEDIA INTEGRATION: Clickable medical links for non-clinical users.
- TEMPORAL SMOOTHING: Debouncing to eliminate AI hallucinations.
- VISUAL EVIDENCE CAPTURE: Snaps photos of faults and embeds them.
"""

import argparse
import math
import os
import sys
import time
import base64
import webbrowser
from collections import deque
from typing import Dict, List, Optional, Tuple, Set, Any

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("ERROR: mediapipe is not installed. Run: pip install mediapipe")
    sys.exit(1)

# ==========================================================================
# SECTION 1: POSE LANDMARK CONSTANTS & WIKIPEDIA DATABASE
# ==========================================================================
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20      
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32  

POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST), (LEFT_SHOULDER, LEFT_HIP), 
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP), (LEFT_HIP, LEFT_KNEE), 
    (LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT_INDEX), (RIGHT_ANKLE, RIGHT_HEEL), 
    (RIGHT_HEEL, RIGHT_FOOT_INDEX), (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR),
]

ANGLE_DEFINITIONS: Dict[str, Tuple[int, int, int]] = {
    "Left Elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "Right Elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "Left Wrist": (LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX),
    "Right Wrist": (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX),
    "Left Shoulder": (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    "Right Shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    "Left Hip": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "Right Hip": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    "Left Knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "Right Knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "Left Ankle": (LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX),
    "Right Ankle": (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX),
}

FAULT_LANDMARK_MAP = {
    "Lateral Head Tilt": [NOSE, LEFT_EAR, RIGHT_EAR],
    "Forward Neck Posture": [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER],
    "Uneven Shoulders": [LEFT_SHOULDER, RIGHT_SHOULDER],
    "Elbow Flare": [LEFT_ELBOW, RIGHT_ELBOW],
    "Left Wrist Collapsing": [LEFT_WRIST, LEFT_INDEX],
    "Right Wrist Collapsing": [RIGHT_WRIST, RIGHT_INDEX],
    "Uneven Hips": [LEFT_HIP, RIGHT_HIP],
    "Knee Valgus": [LEFT_KNEE, RIGHT_KNEE],
    "Ankle Over-Pronation": [LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX],
    "Excessive Forward Lean": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "Spinal Torsion / Twisting": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "Text Neck (Smartphone Hunch)": [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER],
    "Locked Knees (Hyperextension)": [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE],
    "Hazardous Bending (Stoop Lift)": [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE]
}

# WIKIPEDIA INTEGRATED DATABASE
FAULT_DETAILS = {
    "Lateral Head Tilt": {"meaning": "Head leaning to one side.", "fix": "Keep gaze level.", "injury": "Cervical Strain", "url": "https://en.wikipedia.org/wiki/Neck_pain"},
    "Forward Neck Posture": {"meaning": "Head jutting forward past shoulders.", "fix": "Tuck chin, pull head back.", "injury": "Forward Head Posture", "url": "https://en.wikipedia.org/wiki/Forward_head_posture"},
    "Uneven Shoulders": {"meaning": "One shoulder dropping lower.", "fix": "Level shoulders, engage core.", "injury": "Shoulder Impingement", "url": "https://en.wikipedia.org/wiki/Shoulder_impingement_syndrome"},
    "Elbow Flare": {"meaning": "Elbows pointing too far outward.", "fix": "Tuck elbows closer to body.", "injury": "Rotator Cuff Tears", "url": "https://en.wikipedia.org/wiki/Rotator_cuff_tear"},
    "Left Wrist Collapsing": {"meaning": "Left wrist bending dangerously backward.", "fix": "Keep left wrist stacked straight.", "injury": "Wrist Sprain", "url": "https://en.wikipedia.org/wiki/Sprain"},
    "Right Wrist Collapsing": {"meaning": "Right wrist bending dangerously backward.", "fix": "Keep right wrist stacked straight.", "injury": "Wrist Sprain", "url": "https://en.wikipedia.org/wiki/Sprain"},
    "Uneven Hips": {"meaning": "One side of pelvis hiked up.", "fix": "Level pelvis and engage core.", "injury": "SI Joint Dysfunction", "url": "https://en.wikipedia.org/wiki/Sacroiliac_joint_dysfunction"},
    "Knee Valgus": {"meaning": "Knees caving inward toward each other.", "fix": "Push knees outward to align with toes.", "injury": "Genu Valgum (Knock-knees)", "url": "https://en.wikipedia.org/wiki/Genu_valgum"},
    "Ankle Over-Pronation": {"meaning": "Feet flared outward (duck feet).", "fix": "Point toes straight forward.", "injury": "Foot Pronation", "url": "https://en.wikipedia.org/wiki/Pronation_of_the_foot"},
    "Excessive Forward Lean": {"meaning": "Chest is caving and torso is leaning too far forward.", "fix": "Keep chest up and hinge at the hips.", "injury": "Low Back Pain", "url": "https://en.wikipedia.org/wiki/Low_back_pain"},
    "Spinal Torsion / Twisting": {"meaning": "Torso is rotating unevenly under a heavy load.", "fix": "Brace core and lift symmetrically.", "injury": "Muscle Strain", "url": "https://en.wikipedia.org/wiki/Strain_(injury)"},
    "Text Neck (Smartphone Hunch)": {"meaning": "Head tilted sharply downward for an extended period.", "fix": "Bring screens up to eye level.", "injury": "Text Neck", "url": "https://en.wikipedia.org/wiki/Forward_head_posture"},
    "Locked Knees (Hyperextension)": {"meaning": "Standing with knees bent completely backward.", "fix": "Keep a micro-bend in the knees when standing.", "injury": "Genu Recurvatum", "url": "https://en.wikipedia.org/wiki/Genu_recurvatum"},
    "Hazardous Bending (Stoop Lift)": {"meaning": "Bending from the lower back instead of dropping the hips.", "fix": "Bend your knees and use your legs to lift.", "injury": "Spinal Disc Herniation", "url": "https://en.wikipedia.org/wiki/Spinal_disc_herniation"}
}

VISIBILITY_THRESHOLD = 0.75
PRESENCE_THRESHOLD = 0.75
FRAME_MARGIN = 0.05

# ==========================================================================
# SECTION 2: 3D SPATIAL MATHEMATICS
# ==========================================================================
def is_landmark_reliable(lm: Any) -> bool:
    if not lm: return False
    if getattr(lm, "visibility", 0.0) < VISIBILITY_THRESHOLD: return False
    if getattr(lm, "presence", 0.0) < PRESENCE_THRESHOLD: return False
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if x is None or y is None: return False
    if not (-FRAME_MARGIN <= x <= 1 + FRAME_MARGIN): return False
    if not (-FRAME_MARGIN <= y <= 1 + FRAME_MARGIN): return False
    return True

def get_landmark(landmarks: Any, idx: int) -> Any:
    if landmarks is None or idx < 0 or idx >= len(landmarks): return None
    return landmarks[idx]

def calculate_angle_3d(a: Any, b: Any, c: Any) -> Optional[float]:
    if not all(is_landmark_reliable(pt) for pt in (a, b, c)): return None
    try:
        ba = (a.x - b.x, a.y - b.y, getattr(a, 'z', 0) - getattr(b, 'z', 0))
        bc = (c.x - b.x, c.y - b.y, getattr(c, 'z', 0) - getattr(b, 'z', 0))
        dot_product = sum(i * j for i, j in zip(ba, bc))
        mag_ba = math.sqrt(sum(i**2 for i in ba))
        mag_bc = math.sqrt(sum(i**2 for i in bc))
        if mag_ba < 1e-6 or mag_bc < 1e-6: return None
        cos_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))
    except: return None

def midpoint(a: Any, b: Any) -> Optional[Tuple[float, float]]:
    if not (is_landmark_reliable(a) and is_landmark_reliable(b)): return None
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

# ==========================================================================
# SECTION 3: KINEMATIC MOVEMENT & FULL-BODY FORM ANALYZER
# ==========================================================================
class MovementAnalyzer:
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.angle_history: Dict[str, deque] = {name: deque(maxlen=window_size) for name in ANGLE_DEFINITIONS}
        self.torso_mid_history: deque = deque(maxlen=window_size)
        self.torso_angle_history: deque = deque(maxlen=window_size)
        
        self.total_frames_processed = 0
        self.joint_stats: Dict[str, Dict[str, Any]] = {
            name: {"min": 999.0, "max": 0.0, "sum": 0.0, "count": 0} for name in ANGLE_DEFINITIONS
        }
        
        self.logged_events: List[Dict[str, Any]] = []
        self._last_event_time = -5.0
        
        self.detected_form_faults: Set[str] = set()
        self.form_fault_timestamps: Dict[str, List[float]] = {}
        self.snapshot_evidence: Dict[str, List[Tuple[float, str]]] = {}
        self._last_snapshot_times: Dict[str, float] = {}

        self.active_faults: List[str] = []  
        self.current_confidence = 0.0
        self.global_confidence_sum = 0.0

        self.fault_counters: Dict[str, int] = {fault: 0 for fault in FAULT_DETAILS.keys()}
        self.DEBOUNCE_FRAMES = 10  

        self.FALL_TORSO_ANGLE_DEG = 50.0     
        self.FALL_DROP_THRESHOLD = 0.10      
        self.ANKLE_COLLAPSE_RATE = 30.0      

    def analyze_full_body_form(self, landmarks: Any) -> List[str]:
        raw_faults = []
        if not landmarks: return raw_faults
        def h_dist(a, b): return abs(landmarks[a].x - landmarks[b].x)
        def v_dist(a, b): return abs(landmarks[a].y - landmarks[b].y)
        def is_rel(idx): return is_landmark_reliable(get_landmark(landmarks, idx))

        if is_rel(LEFT_EAR) and is_rel(RIGHT_EAR):
            if v_dist(LEFT_EAR, RIGHT_EAR) > 0.05: raw_faults.append("Lateral Head Tilt")
        if is_rel(LEFT_EAR) and is_rel(LEFT_SHOULDER):
            if getattr(landmarks[LEFT_EAR], 'z', 0) < (getattr(landmarks[LEFT_SHOULDER], 'z', 0) - 0.18): raw_faults.append("Forward Neck Posture")
        if is_rel(LEFT_SHOULDER) and is_rel(RIGHT_SHOULDER):
            if v_dist(LEFT_SHOULDER, RIGHT_SHOULDER) > 0.06: raw_faults.append("Uneven Shoulders")
        if is_rel(NOSE) and is_rel(LEFT_EAR) and is_rel(RIGHT_EAR):
            avg_ear_y = (landmarks[LEFT_EAR].y + landmarks[RIGHT_EAR].y) / 2.0
            if landmarks[NOSE].y > (avg_ear_y + 0.05): raw_faults.append("Text Neck (Smartphone Hunch)")
        if is_rel(LEFT_ELBOW) and is_rel(RIGHT_ELBOW) and is_rel(LEFT_SHOULDER) and is_rel(RIGHT_SHOULDER):
            if h_dist(LEFT_ELBOW, RIGHT_ELBOW) > (h_dist(LEFT_SHOULDER, RIGHT_SHOULDER) * 1.8): raw_faults.append("Elbow Flare")
        for side, e_idx, w_idx, i_idx in [("Left", LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX), ("Right", RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX)]:
            if is_rel(e_idx) and is_rel(w_idx) and is_rel(i_idx):
                angle = calculate_angle_3d(landmarks[e_idx], landmarks[w_idx], landmarks[i_idx])
                if angle and angle < 120: raw_faults.append(f"{side} Wrist Collapsing")
        if is_rel(LEFT_SHOULDER) and is_rel(LEFT_HIP):
            if getattr(landmarks[LEFT_SHOULDER], 'z', 0) < (getattr(landmarks[LEFT_HIP], 'z', 0) - 0.35): raw_faults.append("Excessive Forward Lean")
        if is_rel(LEFT_SHOULDER) and is_rel(RIGHT_SHOULDER) and is_rel(LEFT_HIP) and is_rel(RIGHT_HIP):
            shoulder_z_diff = abs(getattr(landmarks[LEFT_SHOULDER], 'z', 0) - getattr(landmarks[RIGHT_SHOULDER], 'z', 0))
            hip_z_diff = abs(getattr(landmarks[LEFT_HIP], 'z', 0) - getattr(landmarks[RIGHT_HIP], 'z', 0))
            if shoulder_z_diff > (hip_z_diff + 0.15): raw_faults.append("Spinal Torsion / Twisting")
        if is_rel(LEFT_SHOULDER) and is_rel(LEFT_HIP) and is_rel(LEFT_KNEE):
            if abs(landmarks[LEFT_SHOULDER].y - landmarks[LEFT_HIP].y) < 0.15:
                ang = calculate_angle_3d(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
                if ang and ang > 140: raw_faults.append("Hazardous Bending (Stoop Lift)")
        if is_rel(LEFT_HIP) and is_rel(RIGHT_HIP):
            if v_dist(LEFT_HIP, RIGHT_HIP) > 0.06: raw_faults.append("Uneven Hips")
        if is_rel(LEFT_KNEE) and is_rel(RIGHT_KNEE) and is_rel(LEFT_ANKLE) and is_rel(RIGHT_ANKLE):
            if h_dist(LEFT_KNEE, RIGHT_KNEE) < (h_dist(LEFT_ANKLE, RIGHT_ANKLE) * 0.7): raw_faults.append("Knee Valgus")
        if is_rel(LEFT_FOOT_INDEX) and is_rel(RIGHT_FOOT_INDEX) and is_rel(LEFT_HEEL) and is_rel(RIGHT_HEEL):
            if h_dist(LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX) > (h_dist(LEFT_HEEL, RIGHT_HEEL) * 1.8): raw_faults.append("Ankle Over-Pronation")
        for side, h_idx, k_idx, a_idx in [("Left", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), ("Right", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]:
            if is_rel(h_idx) and is_rel(k_idx) and is_rel(a_idx):
                angle = calculate_angle_3d(landmarks[h_idx], landmarks[k_idx], landmarks[a_idx])
                if angle and angle > 175: raw_faults.append("Locked Knees (Hyperextension)")
        return raw_faults

    def update(self, landmarks: Any, current_timestamp_sec: float) -> None:
        if not landmarks: 
            self.current_confidence = 0.0
            return
        
        self.total_frames_processed += 1
        tracked_lms = [lm for lm in landmarks if getattr(lm, 'visibility', 0.0) > 0.5]
        if not tracked_lms:
            self.current_confidence = 0.0
        else:
            avg_vis = sum(getattr(lm, 'visibility', 0.0) for lm in tracked_lms) / len(tracked_lms)
            body_ratio = len(tracked_lms) / len(landmarks)
            self.current_confidence = (avg_vis * 0.7 + body_ratio * 0.3) * 100.0
            
        self.global_confidence_sum += self.current_confidence

        for name, (a, b, c) in ANGLE_DEFINITIONS.items():
            angle = calculate_angle_3d(get_landmark(landmarks, a), get_landmark(landmarks, b), get_landmark(landmarks, c))
            self.angle_history[name].append(angle)
            if angle is not None:
                st = self.joint_stats[name]
                st["min"], st["max"] = min(st["min"], angle), max(st["max"], angle)
                st["sum"] += angle
                st["count"] += 1

        l_sh, r_sh = get_landmark(landmarks, LEFT_SHOULDER), get_landmark(landmarks, RIGHT_SHOULDER)
        l_hip, r_hip = get_landmark(landmarks, LEFT_HIP), get_landmark(landmarks, RIGHT_HIP)
        sh_mid, hip_mid = midpoint(l_sh, r_sh), midpoint(l_hip, r_hip)
        if sh_mid and hip_mid:
            self.torso_mid_history.append(((sh_mid[0] + hip_mid[0]) / 2.0, (sh_mid[1] + hip_mid[1]) / 2.0))
            self.torso_angle_history.append(math.degrees(math.atan2(abs(hip_mid[0] - sh_mid[0]), abs(hip_mid[1] - sh_mid[1]) + 1e-6)))
        else:
            self.torso_mid_history.append(None)
            self.torso_angle_history.append(None)

        raw_faults = self.analyze_full_body_form(landmarks)
        for fault_name in self.fault_counters.keys():
            if fault_name in raw_faults: self.fault_counters[fault_name] += 1
            else: self.fault_counters[fault_name] = 0 
                
        self.active_faults = [f for f, count in self.fault_counters.items() if count >= self.DEBOUNCE_FRAMES]
        
        for fault in self.active_faults:
            self.detected_form_faults.add(fault)
            if fault not in self.form_fault_timestamps: self.form_fault_timestamps[fault] = []
            timestamps = self.form_fault_timestamps[fault]
            if not timestamps or (current_timestamp_sec - timestamps[-1] > 3.0):
                timestamps.append(current_timestamp_sec)

    def fall_status(self, current_time_sec: float) -> str:
        positions = [p for p in self.torso_mid_history if p is not None]
        torso_angles = [a for a in self.torso_angle_history if a is not None]
        if len(positions) < 5 or not torso_angles: return "NOMINAL"
        vertical_drop = positions[-1][1] - positions[0][1] 
        if vertical_drop >= self.FALL_DROP_THRESHOLD and torso_angles[-1] >= self.FALL_TORSO_ANGLE_DEG:
            collapse_signals = 0
            for side in ["Left", "Right"]:
                ankle_hist = [a for a in self.angle_history[f"{side} Ankle"] if a is not None]
                if len(ankle_hist) >= 2 and abs(ankle_hist[-1] - ankle_hist[0]) > self.ANKLE_COLLAPSE_RATE: collapse_signals += 1
            state = "CRITICAL: KINEMATIC COLLAPSE" if collapse_signals >= 1 else "WARNING: RAPID DROP"
            if current_time_sec - self._last_event_time > 2.5:
                self.logged_events.append({"timestamp_sec": current_time_sec, "type": state})
                self._last_event_time = current_time_sec
            return state
        return "NOMINAL"

    def should_take_snapshot(self, fault: str, current_time_sec: float) -> bool:
        last_time = self._last_snapshot_times.get(fault, -10.0)
        if current_time_sec - last_time > 5.0:
            self._last_snapshot_times[fault] = current_time_sec
            return True
        return False

    def save_snapshot(self, fault: str, b64_img: str, current_time_sec: float):
        if fault not in self.snapshot_evidence: self.snapshot_evidence[fault] = []
        self.snapshot_evidence[fault].append((current_time_sec, b64_img))

    def print_final_report(self, duration_sec: float, source_name: str) -> None:
        avg_fps = (self.total_frames_processed / duration_sec) if duration_sec > 0 else 0.0
        sys_acc = (self.global_confidence_sum / self.total_frames_processed) if self.total_frames_processed else 0.0

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 40px; line-height: 1.6; max-width: 1000px; margin: auto; }}
                h1 {{ color: #00ffcc; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
                h2 {{ color: #58a6ff; margin-top: 40px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
                .card {{ background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
                .fault-title {{ color: #ff7b72; font-size: 20px; font-weight: bold; margin-bottom: 10px; }}
                .link-btn {{ display: inline-block; background-color: #238636; color: white; padding: 8px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin-top: 15px; }}
                .link-btn:hover {{ background-color: #2ea043; }}
                .timestamp {{ background-color: #1f6feb; color: white; padding: 4px 10px; border-radius: 12px; font-size: 13px; margin-right: 5px; font-weight: bold; display: inline-block; margin-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #30363d; }}
                th {{ background-color: #21262d; color: #c9d1d9; }}
                .snapshot-container {{ display: flex; gap: 15px; overflow-x: auto; padding: 10px 0; margin-top: 15px; border-top: 1px dashed #30363d; }}
                .snapshot-item {{ flex: 0 0 auto; text-align: center; }}
                .snapshot-img {{ height: 250px; border-radius: 8px; border: 2px solid #ff7b72; box-shadow: 0 0 10px rgba(255,123,114,0.3); }}
            </style>
        </head>
        <body>
            <h1>PlaySafe AI : Visual Post-Session Report</h1>
            <p><strong>Source Video:</strong> {source_name} &nbsp;|&nbsp; <strong>AI Confidence Score:</strong> {sys_acc:.1f}%</p>
            <h2>Detected Biomechanical Faults (With Photo Evidence)</h2>
        """
        if self.detected_form_faults:
            for fault in self.detected_form_faults:
                details = FAULT_DETAILS.get(fault, {"meaning": "Unknown", "fix": "Unknown", "injury": "Unknown", "url": "#"})
                snapshots_html = ""
                if fault in self.snapshot_evidence:
                    snapshots_html += "<div class='snapshot-container'>"
                    for ts, b64 in self.snapshot_evidence[fault]:
                        time_str = f"{int(ts//60):02d}:{ts%60:05.2f}"
                        snapshots_html += f"<div class='snapshot-item'><div class='timestamp'>{time_str}</div><br><img class='snapshot-img' src='data:image/jpeg;base64,{b64}'></div>"
                    snapshots_html += "</div>"
                html += f"""
                <div class="card">
                    <div class="fault-title">⚠️ {fault}</div>
                    <p><strong>What it is:</strong> {details['meaning']}</p>
                    <p><strong>How to fix:</strong> {details['fix']}</p>
                    {snapshots_html}
                    <a href="{details['url']}" target="_blank" class="link-btn">Learn about {details['injury']} &rarr;</a>
                </div>
                """
        else:
            html += "<div class='card'><p style='color:#3fb950;'>✅ Excellent form! No biomechanical faults detected.</p></div>"
        html += "<h2>Range of Motion (ROM)</h2><table><tr><th>Joint</th><th>Min</th><th>Max</th><th>Avg</th></tr>"
        for name, st in self.joint_stats.items():
            if st["count"] > 0: html += f"<tr><td>{name}</td><td>{int(st['min'])}&deg;</td><td>{int(st['max'])}&deg;</td><td>{int(st['sum']/st['count'])}&deg;</td></tr>"
            else: html += f"<tr><td>{name}</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>"
        html += "</table></body></html>"
        try:
            with open("PlaySafe_Interactive_Report.html", "w", encoding="utf-8") as f: f.write(html)
            webbrowser.open(f"file://{os.path.abspath('PlaySafe_Interactive_Report.html')}")
        except: pass

# ==========================================================================
# SECTION 4: CYBERNETIC UI (OPEN-CV)
# ==========================================================================
class UIDrawer:
    NEON_CYAN = (255, 220, 0)
    NEON_GREEN = (0, 255, 100)
    CRIMSON_RED = (20, 20, 255)
    DARK_BG = (15, 15, 18)
    PANEL_BG = (25, 25, 30)
    WHITE_GLOW = (240, 240, 255)

    @staticmethod
    def capture_fault_snapshot(frame: np.ndarray, landmarks: Any, fault: str) -> str:
        if not landmarks: return ""
        snap_img = frame.copy()
        h, w = snap_img.shape[:2]
        for idx in FAULT_LANDMARK_MAP.get(fault, []):
            lm = get_landmark(landmarks, idx)
            if is_landmark_reliable(lm):
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(snap_img, (px, py), 25, UIDrawer.CRIMSON_RED, 4, cv2.LINE_AA)
                cv2.circle(snap_img, (px, py), 6, UIDrawer.WHITE_GLOW, -1, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', snap_img)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def add_tech_overlays(frame: np.ndarray, time_sec: float) -> np.ndarray:
        """ Adds a futuristic tactical grid and a scanning laser to the camera feed """
        h, w = frame.shape[:2]
        # Tactical Grid
        for i in range(0, w, 60): cv2.line(frame, (i, 0), (i, h), (40, 50, 40), 1)
        for i in range(0, h, 60): cv2.line(frame, (0, i), (w, i), (40, 50, 40), 1)
        
        # Dynamic Scanning Laser
        scan_y = int(((math.sin(time_sec * 2.5) + 1) / 2) * h)
        cv2.line(frame, (0, scan_y), (w, scan_y), UIDrawer.NEON_CYAN, 1)
        
        # Corner Brackets
        L, T = 30, 3
        cv2.line(frame, (10, 10), (10+L, 10), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (10, 10), (10, 10+L), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (w-10, 10), (w-10-L, 10), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (w-10, 10), (w-10, 10+L), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (10, h-10), (10+L, h-10), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (10, h-10), (10, h-10-L), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (w-10, h-10), (w-10-L, h-10), UIDrawer.NEON_CYAN, T)
        cv2.line(frame, (w-10, h-10), (w-10, h-10-L), UIDrawer.NEON_CYAN, T)
        
        return frame

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: Any, state: str) -> np.ndarray:
        if not landmarks: return frame
        h, w = frame.shape[:2]
        px = lambda lm: (int(lm.x * w), int(lm.y * h))
        base_color = UIDrawer.CRIMSON_RED if "CRITICAL" in state else UIDrawer.NEON_CYAN

        # Create the "Bloom" / Holographic Glow Effect
        overlay = np.zeros_like(frame)
        for idx_a, idx_b in POSE_CONNECTIONS:
            lm_a, lm_b = get_landmark(landmarks, idx_a), get_landmark(landmarks, idx_b)
            if is_landmark_reliable(lm_a) and is_landmark_reliable(lm_b):
                cv2.line(overlay, px(lm_a), px(lm_b), base_color, 8, cv2.LINE_AA)
        
        # Blur the overlay to make it glow, then blend it with the frame
        blur = cv2.GaussianBlur(overlay, (21, 21), 0)
        frame = cv2.addWeighted(frame, 1.0, blur, 0.8, 0)

        # Draw the crisp core skeleton on top
        for idx_a, idx_b in POSE_CONNECTIONS:
            lm_a, lm_b = get_landmark(landmarks, idx_a), get_landmark(landmarks, idx_b)
            if is_landmark_reliable(lm_a) and is_landmark_reliable(lm_b):
                cv2.line(frame, px(lm_a), px(lm_b), UIDrawer.WHITE_GLOW, 2, cv2.LINE_AA)
        for idx in range(len(landmarks)):
            lm = get_landmark(landmarks, idx)
            if is_landmark_reliable(lm):
                cv2.circle(frame, px(lm), 4, base_color, -1, cv2.LINE_AA)
                cv2.circle(frame, px(lm), 2, UIDrawer.WHITE_GLOW, -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_hud(frame: np.ndarray, analyzer: MovementAnalyzer, fps: float, time_sec: float) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w = 480
        canvas = np.full((h, w + panel_w, 3), UIDrawer.DARK_BG, dtype=np.uint8)
        
        # Add tech grid and scanlines to the video
        frame = UIDrawer.add_tech_overlays(frame, time_sec)
        canvas[:, :w] = frame
        
        # Divider Line
        cv2.line(canvas, (w, 0), (w, h), UIDrawer.NEON_CYAN, 2)
        x0, y = w + 25, 40

        def text(txt: str, color: Tuple[int,int,int]=UIDrawer.WHITE_GLOW, scale: float=0.45, thick: int=1, gap: int=25):
            nonlocal y
            cv2.putText(canvas, txt, (x0, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)
            y += gap
            
        def section_header(txt: str):
            nonlocal y
            cv2.rectangle(canvas, (x0, y - 18), (x0 + panel_w - 50, y + 8), UIDrawer.PANEL_BG, -1)
            cv2.putText(canvas, txt, (x0 + 10, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, UIDrawer.NEON_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (x0, y + 8), (x0 + panel_w - 50, y + 8), UIDrawer.NEON_CYAN, 1)
            y += 35

        # HUD HEADER
        cv2.putText(canvas, "PLAYSAFE OS v10", (x0, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, UIDrawer.NEON_CYAN, 2, cv2.LINE_AA)
        y += 35
        text(f"UPLINK FPS : {fps:.1f}  |  TIME: {time_sec:.1f}s", (180, 180, 180))

        # DYNAMIC CONFIDENCE BAR
        y += 10
        cv2.putText(canvas, "SYS CONFIDENCE:", (x0, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, UIDrawer.WHITE_GLOW, 1, cv2.LINE_AA)
        conf = analyzer.current_confidence
        bar_w = 200
        bar_c = UIDrawer.NEON_GREEN if conf > 75 else (UIDrawer.NEON_CYAN if conf > 50 else UIDrawer.CRIMSON_RED)
        cv2.rectangle(canvas, (x0 + 150, y - 10), (x0 + 150 + bar_w, y + 5), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x0 + 150, y - 10), (x0 + 150 + int((conf/100)*bar_w), y + 5), bar_c, -1)
        cv2.putText(canvas, f"{conf:.1f}%", (x0 + 150 + bar_w + 10, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, bar_c, 1)
        y += 40

        # KINEMATIC STATE
        section_header("CORE KINEMATIC STATE")
        fall_stat = analyzer.fall_status(time_sec)
        bar_color = UIDrawer.CRIMSON_RED if "CRITICAL" in fall_stat else (UIDrawer.NEON_CYAN if "WARNING" in fall_stat else UIDrawer.NEON_GREEN)
        cv2.rectangle(canvas, (x0, y), (x0 + panel_w - 50, y + 30), bar_color, 2)
        cv2.putText(canvas, f"> {fall_stat} <", (x0 + 15, y + 21), cv2.FONT_HERSHEY_DUPLEX, 0.6, bar_color, 1, cv2.LINE_AA)
        y += 60

        # LIVE TELEMETRY
        section_header("LIVE SPATIAL TELEMETRY")
        for part in ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"]:
            l_hist = analyzer.angle_history.get(f"Left {part}", [])
            r_hist = analyzer.angle_history.get(f"Right {part}", [])
            l_str = f"{int(l_hist[-1]):03d}" if len(l_hist) > 0 and l_hist[-1] else "---"
            r_str = f"{int(r_hist[-1]):03d}" if len(r_hist) > 0 and r_hist[-1] else "---"
            text(f"DATALINK [{part.upper():<10}] L: {l_str}  R: {r_str}", UIDrawer.WHITE_GLOW, 0.45)
        y += 15

        # ACTIVE FORM CORRECTIONS
        section_header("ACTIVE FORM ANALYSIS")
        if analyzer.active_faults:
            for fault in analyzer.active_faults:
                fix = FAULT_DETAILS.get(fault, {}).get("fix", "")
                text(f"[WARN] {fault.upper()}", UIDrawer.NEON_CYAN)
                text(f" -> {fix}", (150, 150, 255), 0.4, 1, 25)
        else:
            text("[OK] STATUS: OPTIMAL POSTURE", UIDrawer.NEON_GREEN, 0.5)

        return canvas

# ==========================================================================
# SECTION 5: MAIN ENGINE LOOP
# ==========================================================================
def run_engine(model_path: str, video_path: Optional[str] = None, camera_index: int = 0):
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found at: {model_path}")
        sys.exit(1)

    source_desc = video_path if video_path else f"Webcam (Device {camera_index})"
    cap = cv2.VideoCapture(video_path if video_path else camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open video source.")
        sys.exit(1)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.75,
        min_pose_presence_confidence=0.75,
        min_tracking_confidence=0.75,
    )
    
    analyzer = MovementAnalyzer(window_size=15)
    window_name = "PlaySafe OS - Cybernetic UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 850)

    frame_idx, display_fps = 0, 0.0
    start_time = time.time()
    prev_time = start_time
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("PlaySafe OS running. Press Q to quit.")

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: break
            if not video_path: frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            time_sec = frame_idx / fps_source
            timestamp_ms = int(time_sec * 1000)
            frame_idx += 1

            landmarks = None
            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks: landmarks = result.pose_landmarks[0]
            except Exception: pass

            if landmarks:
                analyzer.update(landmarks, time_sec)
                for fault in analyzer.active_faults:
                    if analyzer.should_take_snapshot(fault, time_sec):
                        b64 = UIDrawer.capture_fault_snapshot(frame, landmarks, fault)
                        if b64: analyzer.save_snapshot(fault, b64, time_sec)

            state = analyzer.fall_status(time_sec)
            ui_frame = UIDrawer.draw_skeleton(frame, landmarks, state)

            now = time.time()
            if now - prev_time > 0:
                display_fps = (display_fps * 0.9) + (0.1 * (1.0 / (now - prev_time)))
            prev_time = now

            canvas = UIDrawer.draw_hud(ui_frame, analyzer, display_fps, time_sec)
            cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break

    total_duration = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()
    
    analyzer.print_final_report(total_duration, source_desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaySafe AI Kinematics Engine")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--model", type=str, default="pose_landmarker_full.task", help="Path to model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    
    args = parser.parse_args()
    model_loc = args.model if os.path.isfile(args.model) else os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    run_engine(model_loc, args.video, args.camera)