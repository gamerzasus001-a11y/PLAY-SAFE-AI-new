"""
camera_test.py
--------------
PlaySafe AI - Advanced Kinematic Movement & Fall Screening MVP.

Features 3D spatial angle calculations, kinematic chain collapse detection 
(Base of Support & Bracing), and production-ready type hinting.

USAGE
-----
Webcam: python camera_test.py
Video:  python camera_test.py --video path/to/test.mp4
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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
# SECTION 1: POSE LANDMARK CONSTANTS & CONFIGURATION
# ==========================================================================
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20      # Crucial for Wrist Angle
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32  # Crucial for Ankle Angle

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

# 3-Point Joint Definitions: (Proximal, Vertex, Distal)
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

# Thresholds
VISIBILITY_THRESHOLD = 0.60
PRESENCE_THRESHOLD = 0.60
FRAME_MARGIN = 0.05


# ==========================================================================
# SECTION 2: 3D SPATIAL MATHEMATICS
# ==========================================================================
def is_landmark_reliable(lm: Any) -> bool:
    """Validates landmark occlusion, presence, and frame bounds."""
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
    """
    Computes the 3D angle at vertex b using the dot product of vectors ba and bc.
    Leverages MediaPipe's Z-coordinate for true spatial awareness.
    """
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
# SECTION 3: KINEMATIC MOVEMENT ANALYZER
# ==========================================================================
class MovementAnalyzer:
    """Evaluates biomechanics over a rolling window to detect kinematic collapse."""
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.angle_history: Dict[str, deque] = {name: deque(maxlen=window_size) for name in ANGLE_DEFINITIONS}
        self.torso_mid_history: deque = deque(maxlen=window_size)
        self.torso_angle_history: deque = deque(maxlen=window_size)
        
        # SIH-Tuned Kinematic Thresholds
        self.SUDDEN_CHANGE_DEG = 35.0
        self.ASYMMETRY_DEG = 25.0
        self.TORSO_INCLINE_CAUTION_DEG = 45.0
        
        # Advanced Fall Detection (Kinematic Collapse)
        self.FALL_TORSO_ANGLE_DEG = 50.0     
        self.FALL_DROP_THRESHOLD = 0.10      
        self.ANKLE_COLLAPSE_RATE = 30.0      # Rapid dorsiflexion/plantarflexion
        self.WRIST_BRACE_RATE = 40.0         # Throwing hands out to break a fall

    def update(self, landmarks: Any) -> None:
        if not landmarks: return
        
        # Update 3D Joint Angles
        for name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
            angle = calculate_angle_3d(
                get_landmark(landmarks, a_idx), 
                get_landmark(landmarks, b_idx), 
                get_landmark(landmarks, c_idx)
            )
            self.angle_history[name].append(angle)

        # Update Torso Kinematics
        l_sh, r_sh = get_landmark(landmarks, LEFT_SHOULDER), get_landmark(landmarks, RIGHT_SHOULDER)
        l_hip, r_hip = get_landmark(landmarks, LEFT_HIP), get_landmark(landmarks, RIGHT_HIP)
        sh_mid, hip_mid = midpoint(l_sh, r_sh), midpoint(l_hip, r_hip)

        if sh_mid and hip_mid:
            self.torso_mid_history.append(((sh_mid[0] + hip_mid[0]) / 2.0, (sh_mid[1] + hip_mid[1]) / 2.0))
            dx, dy = hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]
            self.torso_angle_history.append(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))
        else:
            self.torso_mid_history.append(None)
            self.torso_angle_history.append(None)

    def current_angle(self, name: str) -> Optional[float]:
        hist = self.angle_history.get(name)
        return hist[-1] if hist and len(hist) > 0 else None

    def joint_status(self, visibility: Dict[str, bool]) -> Dict[str, str]:
        status = {}
        pairs = ["Elbow", "Wrist", "Shoulder", "Hip", "Knee", "Ankle"]
        
        for part in pairs:
            if not visibility.get(part, False):
                status[part] = "Not shown in video"
                continue
                
            left_hist = [a for a in self.angle_history[f"Left {part}"] if a is not None]
            right_hist = [a for a in self.angle_history[f"Right {part}"] if a is not None]

            if left_hist and right_hist:
                if abs(left_hist[-1] - right_hist[-1]) >= self.ASYMMETRY_DEG:
                    status[part] = "Asymmetry detected"
                    continue
            status[part] = "Normal"
        return status

    def fall_status(self, visibility: Dict[str, bool]) -> str:
        """Evaluates 'Kinematic Chain Collapse' combining torso, ankles, and wrists."""
        if not visibility.get("Torso", False): return "Not shown in video"

        positions = [p for p in self.torso_mid_history if p is not None]
        torso_angles = [a for a in self.torso_angle_history if a is not None]

        if len(positions) < 5 or not torso_angles: return "Stable"

        vertical_drop = positions[-1][1] - positions[0][1] 
        is_falling = vertical_drop >= self.FALL_DROP_THRESHOLD and torso_angles[-1] >= self.FALL_TORSO_ANGLE_DEG

        if not is_falling: return "Stable"

        # Check for secondary Kinematic Collapse (Ankles / Wrists)
        collapse_signals = 0
        for side in ["Left", "Right"]:
            ankle_hist = [a for a in self.angle_history[f"{side} Ankle"] if a is not None]
            wrist_hist = [a for a in self.angle_history[f"{side} Wrist"] if a is not None]
            
            if len(ankle_hist) >= 2 and abs(ankle_hist[-1] - ankle_hist[0]) > self.ANKLE_COLLAPSE_RATE:
                collapse_signals += 1
            if len(wrist_hist) >= 2 and abs(wrist_hist[-1] - wrist_hist[0]) > self.WRIST_BRACE_RATE:
                collapse_signals += 1

        if collapse_signals >= 1:
            return "CRITICAL: Kinematic Collapse (Fall)"
        return "Warning: Rapid Torso Drop"

    def get_flagged_parts(self, visibility: Dict[str, bool]) -> Set[str]:
        flagged = set()
        for part, status in self.joint_status(visibility).items():
            if status != "Normal" and status != "Not shown in video": flagged.add(part)
        
        fall = self.fall_status(visibility)
        if "CRITICAL" in fall:
            flagged.update(["Torso", "Hip", "Ankle", "Wrist"])
        elif "Warning" in fall:
            flagged.update(["Torso", "Hip"])
            
        return flagged


# ==========================================================================
# SECTION 4: RENDERING & UI (OpenCV)
# ==========================================================================
class UIDrawer:
    COLOR_OK = (0, 220, 0)
    COLOR_WARN = (0, 100, 255)
    COLOR_CRIT = (0, 0, 255)
    COLOR_BONE = (240, 240, 240)
    COLOR_BG = (20, 20, 25)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CYAN = (255, 200, 0)  # BGR Cyan for headers

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: Any, flagged_parts: Set[str]) -> np.ndarray:
        if not landmarks: return frame
        h, w = frame.shape[:2]
        
        def px(lm: Any) -> Tuple[int, int]: return int(lm.x * w), int(lm.y * h)

        flagged_indices = set()
        for part in flagged_parts:
            flagged_indices.update(BODY_PART_LANDMARKS.get(part, []))

        # Draw bones
        for idx_a, idx_b in POSE_CONNECTIONS:
            lm_a, lm_b = get_landmark(landmarks, idx_a), get_landmark(landmarks, idx_b)
            if is_landmark_reliable(lm_a) and is_landmark_reliable(lm_b):
                color = UIDrawer.COLOR_CRIT if (idx_a in flagged_indices or idx_b in flagged_indices) else UIDrawer.COLOR_BONE
                cv2.line(frame, px(lm_a), px(lm_b), color, 2, cv2.LINE_AA)

        # Draw joints
        for idx in range(len(landmarks)):
            lm = get_landmark(landmarks, idx)
            if is_landmark_reliable(lm):
                color = UIDrawer.COLOR_CRIT if idx in flagged_indices else UIDrawer.COLOR_OK
                cv2.circle(frame, px(lm), 4, color, -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_hud(frame: np.ndarray, analyzer: MovementAnalyzer, visibility: Dict[str, bool], fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_w = 420
        canvas = np.full((h, w + panel_w, 3), UIDrawer.COLOR_BG, dtype=np.uint8)
        canvas[:, :w] = frame

        x0, y = w + 20, 35

        def text(txt: str, color: Tuple[int,int,int]=UIDrawer.COLOR_TEXT, scale: float=0.5, thick: int=1, gap: int=25):
            nonlocal y
            cv2.putText(canvas, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y += gap

        text("PLAYSAFE AI : KINEMATICS", UIDrawer.COLOR_CYAN, 0.7, 2, 40)
        text(f"Engine FPS: {fps:.1f}", (150, 150, 150), 0.45, 1, 30)

        # Fall Status (Hero Metric)
        fall_stat = analyzer.fall_status(visibility)
        f_color = UIDrawer.COLOR_CRIT if "CRITICAL" in fall_stat else (UIDrawer.COLOR_WARN if "Warning" in fall_stat else UIDrawer.COLOR_OK)
        text("KINEMATIC STATE:", UIDrawer.COLOR_CYAN, 0.55, 1, 25)
        text(f"> {fall_stat}", f_color, 0.6, 2, 35)

        # 3D Angles
        text("LIVE 3D JOINT ANGLES", UIDrawer.COLOR_CYAN, 0.55, 1, 25)
        for part in ["Wrist", "Elbow", "Shoulder", "Hip", "Knee", "Ankle"]:
            l_ang = analyzer.current_angle(f"Left {part}")
            r_ang = analyzer.current_angle(f"Right {part}")
            
            l_str = f"{int(l_ang):03d}" if l_ang else "---"
            r_str = f"{int(r_ang):03d}" if r_ang else "---"
            
            status = analyzer.joint_status(visibility).get(part, "")
            color = UIDrawer.COLOR_WARN if "Asym" in status else UIDrawer.COLOR_TEXT
            
            text(f"{part:<10} L: {l_str}  R: {r_str}", color, 0.5, 1, 22)

        return canvas


# ==========================================================================
# SECTION 5: MAIN ENGINE LOOP
# ==========================================================================
def run_engine(model_path: str, video_path: Optional[str] = None, camera_index: int = 0):
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found at: {model_path}")
        sys.exit(1)

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
    window_name = "PlaySafe AI Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1500, 800)

    frame_idx, display_fps, prev_time = 0, 0.0, time.time()
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("PlaySafe AI Engine running. Press Q to quit.")

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: break
            if not video_path: frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_idx / fps_source) * 1000)
            frame_idx += 1

            landmarks = None
            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks: landmarks = result.pose_landmarks[0]
            except Exception: pass

            visibility = {p: False for p in BODY_PART_LANDMARKS}
            if landmarks:
                for part, idx_list in BODY_PART_LANDMARKS.items():
                    flags = [is_landmark_reliable(get_landmark(landmarks, i)) for i in idx_list]
                    visibility[part] = any(flags)
                analyzer.update(landmarks)

            flagged = analyzer.get_flagged_parts(visibility)
            frame = UIDrawer.draw_skeleton(frame, landmarks, flagged)

            now = time.time()
            if now - prev_time > 0:
                display_fps = (display_fps * 0.9) + (0.1 * (1.0 / (now - prev_time)))
            prev_time = now

            canvas = UIDrawer.draw_hud(frame, analyzer, visibility, display_fps)
            cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaySafe AI Kinematics Engine")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--model", type=str, default="pose_landmarker_full.task", help="Path to model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    
    args = parser.parse_args()
    
    model_loc = args.model if os.path.isfile(args.model) else os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    run_engine(model_loc, args.video, args.camera)