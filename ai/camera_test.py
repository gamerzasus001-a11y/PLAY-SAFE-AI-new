"""
camera_test.py
--------------
PlaySafe AI - Advanced Kinematic Movement & Fall Screening MVP.

Features:
- Full-Body Form Analysis (Head, Shoulders, Elbows, Wrists, Hips, Knees, Ankles)
- Form Fault Timestamp Logging for video review
- 3D spatial angle calculations leveraging Z-depth
- Kinematic chain collapse detection (Falls)
- Live Confidence/Occlusion Metric
- High-Tech "Cyber" OpenCV HUD
- Complete Post-Analysis Session Report
"""

import argparse
import math
import os
import sys
import time
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
# SECTION 1: POSE LANDMARK CONSTANTS
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

BODY_PART_LANDMARKS: Dict[str, List[int]] = {
    "Head": [NOSE, LEFT_EAR, RIGHT_EAR],
    "Shoulder": [LEFT_SHOULDER, RIGHT_SHOULDER],
    "Elbow": [LEFT_ELBOW, RIGHT_ELBOW],
    "Wrist": [LEFT_WRIST, RIGHT_WRIST, LEFT_INDEX, RIGHT_INDEX],
    "Torso": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "Hip": [LEFT_HIP, RIGHT_HIP],
    "Knee": [LEFT_KNEE, RIGHT_KNEE],
    "Ankle": [LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX],
}

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

VISIBILITY_THRESHOLD = 0.60
PRESENCE_THRESHOLD = 0.60
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
    except (ValueError, ZeroDivisionError, AttributeError):
        return None

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
        
        # Session Analytics Accumulators
        self.total_frames_processed = 0
        self.joint_stats: Dict[str, Dict[str, Any]] = {
            name: {"min": 999.0, "max": 0.0, "sum": 0.0, "count": 0} for name in ANGLE_DEFINITIONS
        }
        
        # Fall tracking
        self.logged_events: List[Dict[str, Any]] = []
        self._last_event_time = -5.0
        
        # Form tracking with timestamps
        self.detected_form_faults: Dict[str, str] = {}
        self.form_fault_timestamps: Dict[str, List[float]] = {}  # Tracks all timestamps for each fault
        self.active_faults: Dict[str, str] = {}  

        # Confidence Metrics
        self.current_confidence = 0.0
        self.global_confidence_sum = 0.0

        # Fall Thresholds
        self.FALL_TORSO_ANGLE_DEG = 50.0     
        self.FALL_DROP_THRESHOLD = 0.10      
        self.ANKLE_COLLAPSE_RATE = 30.0      
        self.WRIST_BRACE_RATE = 40.0         

    def analyze_full_body_form(self, landmarks: Any, visibility: Dict[str, bool]) -> Dict[str, str]:
        faults = {}
        if not landmarks: return faults

        def h_dist(a, b): return abs(landmarks[a].x - landmarks[b].x)
        def v_dist(a, b): return abs(landmarks[a].y - landmarks[b].y)

        if visibility.get("Head", False):
            if v_dist(LEFT_EAR, RIGHT_EAR) > 0.04:
                faults["Lateral Head Tilt"] = "Keep gaze level to protect cervical spine."
            if getattr(landmarks[LEFT_EAR], 'z', 0) < (getattr(landmarks[LEFT_SHOULDER], 'z', 0) - 0.15):
                faults["Forward Neck Posture"] = "Tuck chin, pull head back over shoulders."
        if visibility.get("Shoulder", False) and v_dist(LEFT_SHOULDER, RIGHT_SHOULDER) > 0.05:
            faults["Uneven Shoulders"] = "Level shoulders to correct core imbalance."
        if visibility.get("Elbow", False) and visibility.get("Shoulder", False):
            if h_dist(LEFT_ELBOW, RIGHT_ELBOW) > (h_dist(LEFT_SHOULDER, RIGHT_SHOULDER) * 1.6):
                faults["Elbow Flare"] = "Tuck elbows closer to body to protect joints."
        if visibility.get("Wrist", False):
            for side, elbow, wrist, index in [("Left", LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX), ("Right", RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX)]:
                angle = calculate_angle_3d(landmarks[elbow], landmarks[wrist], landmarks[index])
                if angle and angle < 140:
                    faults[f"{side} Wrist Collapsing"] = f"Keep {side.lower()} wrist neutral and stacked."
        if visibility.get("Hip", False) and v_dist(LEFT_HIP, RIGHT_HIP) > 0.05:
            faults["Uneven Hips"] = "Level your pelvis to avoid lower back strain."
        if visibility.get("Knee", False) and visibility.get("Ankle", False):
            if h_dist(LEFT_KNEE, RIGHT_KNEE) < (h_dist(LEFT_ANKLE, RIGHT_ANKLE) * 0.6):
                faults["Knee Valgus"] = "Push knees outward to align with toes."
        if visibility.get("Ankle", False):
            if h_dist(LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX) > (h_dist(LEFT_HEEL, RIGHT_HEEL) * 1.6):
                faults["Ankle Over-Pronation"] = "Point toes forward to stop knee shear."
        return faults

    def update(self, landmarks: Any, current_timestamp_sec: float, visibility: Dict[str, bool]) -> None:
        if not landmarks: 
            self.current_confidence = 0.0
            return
        
        self.total_frames_processed += 1

        # Calculate Confidence Score based on Neural Network Probabilities
        vis_scores = [(getattr(lm, 'visibility', 0.0) + getattr(lm, 'presence', 0.0)) / 2.0 for lm in landmarks]
        self.current_confidence = (sum(vis_scores) / len(vis_scores)) * 100.0
        self.global_confidence_sum += self.current_confidence

        for name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
            angle = calculate_angle_3d(get_landmark(landmarks, a_idx), get_landmark(landmarks, b_idx), get_landmark(landmarks, c_idx))
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

        # Update Form Faults and Log Timestamps
        self.active_faults = self.analyze_full_body_form(landmarks, visibility)
        for fault, fix in self.active_faults.items():
            self.detected_form_faults[fault] = fix
            
            # Initialize timestamp list if new fault
            if fault not in self.form_fault_timestamps:
                self.form_fault_timestamps[fault] = []
            
            # Add timestamp if it's the first time, or if 3 seconds have passed (cooldown)
            timestamps = self.form_fault_timestamps[fault]
            if not timestamps or (current_timestamp_sec - timestamps[-1] > 3.0):
                timestamps.append(current_timestamp_sec)

    def fall_status(self, visibility: Dict[str, bool], current_time_sec: float) -> str:
        if not visibility.get("Torso", False): return "SYS: STANDBY"
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

    def print_final_report(self, duration_sec: float, source_name: str) -> None:
        avg_fps = (self.total_frames_processed / duration_sec) if duration_sec > 0 else 0.0
        sys_acc = (self.global_confidence_sum / self.total_frames_processed) if self.total_frames_processed else 0.0

        report_lines = []
        report_lines.append("========================================================================")
        report_lines.append("                  PLAYSAFE AI : KINEMATIC SESSION REPORT                ")
        report_lines.append("========================================================================")
        report_lines.append(f" Source Media       : {source_name}")
        report_lines.append(f" Total Frames       : {self.total_frames_processed}")
        report_lines.append(f" Duration Processed : {duration_sec:.2f} seconds")
        report_lines.append(f" Average FPS        : {avg_fps:.1f}")
        report_lines.append(f" AI Confidence Score: {sys_acc:.1f}% (Based on Hardware Occlusion & Probability)")
        report_lines.append("------------------------------------------------------------------------")
        report_lines.append(" BIOMECHANICAL RANGE OF MOTION (ROM) SUMMARY")
        report_lines.append("------------------------------------------------------------------------")
        report_lines.append(f" {'Joint Name':<16} | {'Min Angle':<10} | {'Max Angle':<10} | {'Avg Angle':<10}")
        report_lines.append(" " + "-"*65)

        for name, st in self.joint_stats.items():
            if st["count"] > 0:
                min_deg, max_deg, avg_deg = f"{int(st['min'])} deg", f"{int(st['max'])} deg", f"{int(st['sum'] / st['count'])} deg"
            else:
                min_deg, max_deg, avg_deg = "N/A", "N/A", "N/A"
            report_lines.append(f" {name:<16} | {min_deg:<10} | {max_deg:<10} | {avg_deg:<10}")

        report_lines.append("------------------------------------------------------------------------")
        report_lines.append(" LOGGED CRITICAL KINEMATIC EVENTS (FALLS)")
        report_lines.append("------------------------------------------------------------------------")
        if self.logged_events:
            for ev in self.logged_events:
                report_lines.append(f"  [!] Timestamp [{int(ev['timestamp_sec'] // 60):02d}:{ev['timestamp_sec'] % 60:05.2f}] -> {ev['type']}")
        else:
            report_lines.append("  [+] No high-confidence fall or impact events detected.")

        report_lines.append("------------------------------------------------------------------------")
        report_lines.append(" BIOMECHANICAL FORM CORRECTION PLAN & TIMESTAMPS")
        report_lines.append("------------------------------------------------------------------------")
        if self.detected_form_faults:
            for fault, fix in self.detected_form_faults.items():
                report_lines.append(f" [FAULT] {fault}")
                report_lines.append(f"   -> [FIX] {fix}")
                
                # Format all timestamps recorded for this specific fault
                timestamps = self.form_fault_timestamps.get(fault, [])
                formatted_ts = [f"{int(t // 60):02d}:{t % 60:05.2f}" for t in timestamps]
                
                # Print timestamps neatly (max 10 per line to avoid messy wrapping)
                ts_str = ", ".join(formatted_ts[:10])
                if len(formatted_ts) > 10: ts_str += f", ... (and {len(formatted_ts)-10} more)"
                
                report_lines.append(f"   -> [OCCURRED AT] {ts_str}")
                report_lines.append("")
        else:
            report_lines.append(" [+] Excellent form! No persistent misalignments detected.")

        report_lines.append("========================================================================")

        report_str = "\n".join(report_lines)
        print("\n" + report_str + "\n")

        try:
            with open("PlaySafe_Kinematic_Report.txt", "w") as f: f.write(report_str)
        except Exception: pass


# ==========================================================================
# SECTION 4: HIGH-TECH CYBERNETIC UI (OpenCV)
# ==========================================================================
class UIDrawer:
    NEON_CYAN = (255, 220, 0)
    NEON_GREEN = (0, 255, 100)
    CRIMSON_RED = (20, 20, 255)
    DARK_BG = (15, 15, 18)
    PANEL_BG = (25, 25, 30)
    WHITE_GLOW = (240, 240, 255)

    @staticmethod
    def draw_target_corners(frame: np.ndarray, landmarks: Any, color: Tuple[int,int,int]):
        if not landmarks: return
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in landmarks if is_landmark_reliable(lm)]
        y_coords = [lm.y * h for lm in landmarks if is_landmark_reliable(lm)]
        if not x_coords or not y_coords: return
        padding, L, T = 30, 25, 2
        min_x, max_x = int(min(x_coords)) - padding, int(max(x_coords)) + padding
        min_y, max_y = int(min(y_coords)) - padding, int(max(y_coords)) + padding
        
        cv2.line(frame, (min_x, min_y), (min_x + L, min_y), color, T)
        cv2.line(frame, (min_x, min_y), (min_x, min_y + L), color, T)
        cv2.line(frame, (max_x, min_y), (max_x - L, min_y), color, T)
        cv2.line(frame, (max_x, min_y), (max_x, min_y + L), color, T)
        cv2.line(frame, (min_x, max_y), (min_x + L, max_y), color, T)
        cv2.line(frame, (min_x, max_y), (min_x, max_y - L), color, T)
        cv2.line(frame, (max_x, max_y), (max_x - L, max_y), color, T)
        cv2.line(frame, (max_x, max_y), (max_x, max_y - L), color, T)

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: Any, state: str) -> np.ndarray:
        if not landmarks: return frame
        h, w = frame.shape[:2]
        px = lambda lm: (int(lm.x * w), int(lm.y * h))
        base_color = UIDrawer.CRIMSON_RED if "CRITICAL" in state else UIDrawer.NEON_CYAN

        overlay = frame.copy()
        for idx_a, idx_b in POSE_CONNECTIONS:
            lm_a, lm_b = get_landmark(landmarks, idx_a), get_landmark(landmarks, idx_b)
            if is_landmark_reliable(lm_a) and is_landmark_reliable(lm_b):
                cv2.line(overlay, px(lm_a), px(lm_b), base_color, 4, cv2.LINE_AA)
        
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        for idx_a, idx_b in POSE_CONNECTIONS:
            lm_a, lm_b = get_landmark(landmarks, idx_a), get_landmark(landmarks, idx_b)
            if is_landmark_reliable(lm_a) and is_landmark_reliable(lm_b):
                cv2.line(frame, px(lm_a), px(lm_b), UIDrawer.WHITE_GLOW, 1, cv2.LINE_AA)
        for idx in range(len(landmarks)):
            lm = get_landmark(landmarks, idx)
            if is_landmark_reliable(lm):
                cv2.circle(frame, px(lm), 4, base_color, -1, cv2.LINE_AA)
                cv2.circle(frame, px(lm), 2, UIDrawer.WHITE_GLOW, -1, cv2.LINE_AA)
                
        UIDrawer.draw_target_corners(frame, landmarks, base_color)
        return frame

    @staticmethod
    def draw_hud(frame: np.ndarray, analyzer: MovementAnalyzer, visibility: Dict[str, bool], fps: float, time_sec: float) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w = 480
        
        canvas = np.full((h, w + panel_w, 3), UIDrawer.DARK_BG, dtype=np.uint8)
        canvas[:, :w] = frame
        cv2.line(canvas, (w, 0), (w, h), UIDrawer.NEON_CYAN, 2)
        x0, y = w + 25, 40

        def text(txt: str, color: Tuple[int,int,int]=UIDrawer.WHITE_GLOW, scale: float=0.5, thick: int=1, gap: int=25):
            nonlocal y
            cv2.putText(canvas, txt, (x0, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)
            y += gap
            
        def section_header(txt: str):
            nonlocal y
            cv2.rectangle(canvas, (x0, y - 18), (x0 + panel_w - 50, y + 8), UIDrawer.PANEL_BG, -1)
            cv2.putText(canvas, txt, (x0 + 10, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, UIDrawer.NEON_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (x0, y + 8), (x0 + panel_w - 50, y + 8), UIDrawer.NEON_CYAN, 1)
            y += 35

        # HUD HEADER
        text("PLAYSAFE OS v4.5", UIDrawer.NEON_CYAN, 0.8, 2, 30)
        text(f"UPLINK FPS : {fps:.1f}   |   SYS CONFIDENCE: {analyzer.current_confidence:.1f}%", (180, 180, 180), 0.45, 1, 40)

        # KINEMATIC STATE
        section_header("CORE KINEMATIC STATE")
        fall_stat = analyzer.fall_status(visibility, time_sec)
        bar_color = UIDrawer.CRIMSON_RED if "CRITICAL" in fall_stat else (UIDrawer.NEON_CYAN if "WARNING" in fall_stat else UIDrawer.NEON_GREEN)
        cv2.rectangle(canvas, (x0, y), (x0 + panel_w - 50, y + 30), bar_color, 2)
        cv2.putText(canvas, f"> {fall_stat} <", (x0 + 15, y + 21), cv2.FONT_HERSHEY_DUPLEX, 0.6, bar_color, 1, cv2.LINE_AA)
        y += 60

        # LIVE TELEMETRY
        section_header("LIVE SPATIAL TELEMETRY")
        for part in ["Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"]:
            l_ang = analyzer.angle_history[f"Left {part}"][-1] if analyzer.angle_history[f"Left {part}"] else None
            r_ang = analyzer.angle_history[f"Right {part}"][-1] if analyzer.angle_history[f"Right {part}"] else None
            
            l_str = f"{int(l_ang):03d}" if l_ang is not None else "---"
            r_str = f"{int(r_ang):03d}" if r_ang is not None else "---"
            
            text(f"[{part.upper():<10}]   L: {l_str}   R: {r_str}", UIDrawer.WHITE_GLOW, 0.5, 1, 24)
        y += 15

        # ACTIVE FORM CORRECTIONS
        section_header("ACTIVE FORM ANALYSIS")
        if analyzer.active_faults:
            for fault, fix in analyzer.active_faults.items():
                text(f"WARN: {fault.upper()}", UIDrawer.NEON_CYAN, 0.5, 1, 20)
                text(f" -> {fix}", (150, 150, 255), 0.4, 1, 25)
        else:
            text("STATUS: OPTIMAL POSTURE", UIDrawer.NEON_GREEN, 0.5, 1, 30)

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
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    
    analyzer = MovementAnalyzer(window_size=15)
    window_name = "PlaySafe OS - Biomechanical UI"
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

            visibility = {p: False for p in BODY_PART_LANDMARKS}
            if landmarks:
                for part, idx_list in BODY_PART_LANDMARKS.items():
                    visibility[part] = any(is_landmark_reliable(get_landmark(landmarks, i)) for i in idx_list)
                analyzer.update(landmarks, time_sec, visibility)

            state = analyzer.fall_status(visibility, time_sec)
            frame = UIDrawer.draw_skeleton(frame, landmarks, state)

            now = time.time()
            if now - prev_time > 0:
                display_fps = (display_fps * 0.9) + (0.1 * (1.0 / (now - prev_time)))
            prev_time = now

            canvas = UIDrawer.draw_hud(frame, analyzer, visibility, display_fps, time_sec)
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