"""
camera_test.py
--------------
PlaySafe AI - Advanced Kinematic Engine (JARVIS WEB EDITION)
"""
import argparse, math, os, sys, time, base64, webbrowser, threading, queue, json
from collections import deque
from typing import List, Optional, Tuple, Any
import cv2
import numpy as np

try:
    import pyttsx3
except ImportError:
    print("ERROR: pyttsx3 not found. Run: pip install pyttsx3")
    sys.exit(1)

# Bridge to Gemini
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from gemini_analyzer import get_gemini_coach_insight
except ImportError:
    def get_gemini_coach_insight(*args): return "Gemini AI Coach Analysis unavailable locally."

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("ERROR: mediapipe is not installed.")
    sys.exit(1)

# ==========================================================================
# VOICE ASSISTANT ENGINE
# ==========================================================================
class VoiceAssistant:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.last_spoken = {}
        
    def _run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            while True:
                text = self.q.get()
                if text is None: break
                engine.say(text)
                engine.runAndWait()
        except: pass
            
    def speak(self, text, fault_id, current_time):
        if current_time - self.last_spoken.get(fault_id, -10.0) > 7.0:
            self.q.put(text)
            self.last_spoken[fault_id] = current_time

voice_ai = VoiceAssistant()

# ==========================================================================
# CONSTANTS & DICTIONARIES
# ==========================================================================
NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
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
    (RIGHT_HEEL, RIGHT_FOOT_INDEX), (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR)
]

ANGLE_DEFINITIONS = {
    "Left Elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), "Right Elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "Left Wrist": (LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX), "Right Wrist": (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX),
    "Left Shoulder": (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP), "Right Shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    "Left Hip": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), "Right Hip": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    "Left Knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), "Right Knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "Left Ankle": (LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX), "Right Ankle": (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX)
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
    "Severe Spinal Rounding": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE],
    "Excessive Forward Lean": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "Locked Knees (Hyperextension)": [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]
}

FAULT_DETAILS = {
    "Lateral Head Tilt": {"meaning": "Head leaning off vertical axis.", "fix": "Maintain level neck vector.", "wiki": "https://en.wikipedia.org/wiki/Cervical_spine"},
    "Forward Neck Posture": {"meaning": "Cervical spine extending forward beyond shoulders.", "fix": "Retract chin, stack cervical joints over torso.", "wiki": "https://en.wikipedia.org/wiki/Forward_head_posture"},
    "Uneven Shoulders": {"meaning": "Asymmetrical shoulder elevation.", "fix": "Depress scapulae and level clavicle baseline.", "wiki": "https://en.wikipedia.org/wiki/Shoulder_impingement_syndrome"},
    "Elbow Flare": {"meaning": "Humeral abduction flaring outward.", "fix": "Tuck elbows tight to lateral torso plane.", "wiki": "https://en.wikipedia.org/wiki/Tendonitis"},
    "Left Wrist Collapsing": {"meaning": "Left wrist hyperextension under load.", "fix": "Maintain rigid linear wrist alignment.", "wiki": "https://en.wikipedia.org/wiki/Wrist_pain"},
    "Right Wrist Collapsing": {"meaning": "Right wrist hyperextension under load.", "fix": "Maintain rigid linear wrist alignment.", "wiki": "https://en.wikipedia.org/wiki/Wrist_pain"},
    "Uneven Hips": {"meaning": "Pelvic tilt or lateral hip hiking.", "fix": "Level pelvis symmetrically and engage core.", "wiki": "https://en.wikipedia.org/wiki/Pelvic_tilt"},
    "Knee Valgus": {"meaning": "Medial knee collapse inward.", "fix": "Drive knees outward along tracking line of toes.", "wiki": "https://en.wikipedia.org/wiki/Genu_valgum"},
    "Severe Spinal Rounding": {"meaning": "Fishing-rod spine. Torso is bent over but knees are stiff.", "fix": "Drop your hips and bend your knees. Hinge properly—do NOT lift with your back!", "wiki": "https://en.wikipedia.org/wiki/Kyphosis"},
    "Excessive Forward Lean": {"meaning": "Thoracic angle collapsing forward.", "fix": "Maintain upright torso angle.", "wiki": "https://en.wikipedia.org/wiki/Lumbar_hyperlordosis"},
    "Locked Knees (Hyperextension)": {"meaning": "Knee joint pushed past 180 degrees.", "fix": "Maintain micro-flexion at knee joints.", "wiki": "https://en.wikipedia.org/wiki/Genu_recurvatum"}
}

def is_landmark_reliable(lm: Any) -> bool:
    if not lm or getattr(lm, "visibility", 0.0) < 0.65 or getattr(lm, "presence", 0.0) < 0.65: return False
    return lm.x is not None and lm.y is not None and (-0.05 <= lm.x <= 1.05) and (-0.05 <= lm.y <= 1.05)

def get_landmark(landmarks: Any, idx: int) -> Any:
    return landmarks[idx] if landmarks and 0 <= idx < len(landmarks) else None

def calculate_angle_3d(a: Any, b: Any, c: Any) -> Optional[float]:
    if not all(is_landmark_reliable(pt) for pt in (a, b, c)): return None
    try:
        ba = (a.x - b.x, a.y - b.y, getattr(a, 'z', 0) - getattr(b, 'z', 0))
        bc = (c.x - b.x, c.y - b.y, getattr(c, 'z', 0) - getattr(b, 'z', 0))
        dot_product = sum(i * j for i, j in zip(ba, bc))
        mag_ba, mag_bc = math.sqrt(sum(i**2 for i in ba)), math.sqrt(sum(i**2 for i in bc))
        if mag_ba < 1e-6 or mag_bc < 1e-6: return None
        return math.degrees(math.acos(max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))))
    except: return None

class MovementAnalyzer:
    def __init__(self, window_size: int = 15):
        self.angle_history = {name: deque(maxlen=window_size) for name in ANGLE_DEFINITIONS}
        self.joint_stats = {name: {"min": 999.0, "max": 0.0, "sum": 0.0, "count": 0} for name in ANGLE_DEFINITIONS}
        self.active_faults = []
        self.snapshot_evidence = {}
        self._last_snapshot_times = {}
        self.fault_counters = {fault: 0 for fault in FAULT_DETAILS.keys()}
        self.total_frames = 0
        self.global_conf = 0.0
        self.pending_snapshots = set() 

    def analyze_form(self, lms: Any) -> List[str]:
        faults = []
        if not lms: return faults
        hd = lambda a, b: abs(lms[a].x - lms[b].x)
        vd = lambda a, b: abs(lms[a].y - lms[b].y)
        rel = lambda idx: is_landmark_reliable(get_landmark(lms, idx))

        if rel(LEFT_EAR) and rel(RIGHT_EAR) and vd(LEFT_EAR, RIGHT_EAR) > 0.05: faults.append("Lateral Head Tilt")
        
        if rel(LEFT_ELBOW) and rel(RIGHT_ELBOW) and rel(LEFT_SHOULDER) and rel(RIGHT_SHOULDER):
            if hd(LEFT_ELBOW, RIGHT_ELBOW) > (hd(LEFT_SHOULDER, RIGHT_SHOULDER) * 1.5): faults.append("Elbow Flare")
            
        for side, e, w, i in [("Left", LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX), ("Right", RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX)]:
            if rel(e) and rel(w) and rel(i):
                ang = calculate_angle_3d(lms[e], lms[w], lms[i])
                if ang and ang < 140: faults.append(f"{side} Wrist Collapsing")
        
        for sh, hi, kn, an in [(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]:
            if rel(sh) and rel(hi) and rel(kn) and rel(an):
                hip_ang = calculate_angle_3d(lms[sh], lms[hi], lms[kn])
                knee_ang = calculate_angle_3d(lms[hi], lms[kn], lms[an])
                if hip_ang and knee_ang:
                    if hip_ang < 95 and knee_ang > 115:
                        faults.append("Severe Spinal Rounding")
                        break 

        if rel(LEFT_KNEE) and rel(RIGHT_KNEE) and rel(LEFT_ANKLE) and rel(RIGHT_ANKLE):
            if hd(LEFT_KNEE, RIGHT_KNEE) < (hd(LEFT_ANKLE, RIGHT_ANKLE) * 0.8): faults.append("Knee Valgus")
            
        return faults

    def update(self, lms: Any, ts: float) -> None:
        if not lms: return
        self.total_frames += 1
        tracked = [lm for lm in lms if getattr(lm, 'visibility', 0.0) > 0.5]
        self.global_conf += (len(tracked) / len(lms)) * 100.0 if tracked else 0.0
        
        for name, (a, b, c) in ANGLE_DEFINITIONS.items():
            ang = calculate_angle_3d(get_landmark(lms, a), get_landmark(lms, b), get_landmark(lms, c))
            self.angle_history[name].append(ang)
            if ang is not None:
                self.joint_stats[name]["min"] = min(self.joint_stats[name]["min"], ang)
                self.joint_stats[name]["max"] = max(self.joint_stats[name]["max"], ang)
                self.joint_stats[name]["sum"] += ang
                self.joint_stats[name]["count"] += 1
        
        raw = self.analyze_form(lms)
        for f in self.fault_counters.keys():
            self.fault_counters[f] = self.fault_counters[f] + 1 if f in raw else 0 
        
        self.active_faults = [f for f, c in self.fault_counters.items() if c >= 5]
        
        for f in self.active_faults:
            voice_ai.speak(f"Warning. {f} detected.", f, ts)
            if ts - self._last_snapshot_times.get(f, -10.0) > 5.0:
                self._last_snapshot_times[f] = ts
                self.pending_snapshots.add(f) 

    def save_snap(self, f: str, b64: str, ts: float):
        if f not in self.snapshot_evidence: 
            self.snapshot_evidence[f] = []
        self.snapshot_evidence[f].append((ts, b64))

    # DYNAMIC SVG GENERATOR
    def get_dynamic_svg(self, fault_name: str) -> str:
        base_grid = '<path d="M 0,20 L 100,20 M 0,40 L 100,40 M 0,60 L 100,60 M 0,80 L 100,80 M 20,0 L 20,100 M 40,0 L 40,100 M 60,0 L 60,100 M 80,0 L 80,100" stroke="#1e293b" stroke-width="1" />'
        
        if "Neck" in fault_name or "Head" in fault_name:
            # Vertical Spine, Head tracking perfectly straight
            return f'<svg viewBox="0 0 100 100">{base_grid}<circle cx="50" cy="25" r="7" stroke="#10b981" stroke-width="3" fill="#080d1a" /><line x1="50" y1="32" x2="50" y2="80" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="50" y1="10" x2="50" y2="90" stroke="#38bdf8" stroke-width="1" stroke-dasharray="4" /></svg>'
            
        elif "Shoulder" in fault_name or "Elbow" in fault_name or "Wrist" in fault_name:
            # Frontal view, level shoulders, arms straight
            return f'<svg viewBox="0 0 100 100">{base_grid}<circle cx="50" cy="20" r="7" stroke="#10b981" stroke-width="3" fill="#080d1a" /><line x1="50" y1="27" x2="50" y2="70" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="30" y1="35" x2="70" y2="35" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="30" y1="35" x2="30" y2="65" stroke="#10b981" stroke-width="3" stroke-linecap="round" /><line x1="70" y1="35" x2="70" y2="65" stroke="#10b981" stroke-width="3" stroke-linecap="round" /><line x1="20" y1="35" x2="80" y2="35" stroke="#38bdf8" stroke-width="1" stroke-dasharray="4" /></svg>'
            
        elif "Knee" in fault_name or "Ankle" in fault_name or "Hip" in fault_name:
            # Lower body, perfect straight tracking legs
            return f'<svg viewBox="0 0 100 100">{base_grid}<line x1="35" y1="20" x2="65" y2="20" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="35" y1="20" x2="35" y2="55" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="65" y1="20" x2="65" y2="55" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><circle cx="35" cy="55" r="3" fill="#38bdf8" /><circle cx="65" cy="55" r="3" fill="#38bdf8" /><line x1="35" y1="55" x2="35" y2="90" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="65" y1="55" x2="65" y2="90" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="35" y1="10" x2="35" y2="95" stroke="#38bdf8" stroke-width="1" stroke-dasharray="4" /><line x1="65" y1="10" x2="65" y2="95" stroke="#38bdf8" stroke-width="1" stroke-dasharray="4" /></svg>'
            
        else: 
            # Spinal / Leaning (Deadlift Hinge view)
            return f'<svg viewBox="0 0 100 100">{base_grid}<circle cx="70" cy="25" r="7" stroke="#10b981" stroke-width="3" fill="#080d1a" /><line x1="68" y1="32" x2="40" y2="60" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="60" y1="40" x2="65" y2="70" stroke="#10b981" stroke-width="3" stroke-linecap="round" /><circle cx="65" cy="70" r="3" fill="#38bdf8" /><line x1="40" y1="60" x2="55" y2="85" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="55" y1="85" x2="55" y2="95" stroke="#10b981" stroke-width="4" stroke-linecap="round" /><line x1="50" y1="95" x2="65" y2="95" stroke="#10b981" stroke-width="3" stroke-linecap="round" /><line x1="75" y1="25" x2="33" y2="67" stroke="#38bdf8" stroke-width="1" stroke-dasharray="4" /></svg>'

    def print_report(self, src: str, out_vid: str, gemini_summary: str = ""):
        acc = (self.global_conf / self.total_frames) if self.total_frames else 0.0
        evts = [{"time": float(ts), "fault": f, "msg": f"Alert. {f} detected."} for f, snaps in self.snapshot_evidence.items() for ts, _ in snaps]
        js = json.dumps(evts)

        video_b64 = ""
        if out_vid and os.path.exists(out_vid):
            try:
                with open(out_vid, "rb") as vf:
                    video_b64 = base64.b64encode(vf.read()).decode('utf-8')
            except Exception as e:
                print(f"Error encoding video to Base64: {e}")

        html = f"""
        <!DOCTYPE html>
        <html><head><title>PlaySafe AI Report</title>
        <style>
            body {{ background:#050a14; color:#e2e8f0; font-family:'Segoe UI', Tahoma, sans-serif; padding:30px; max-width:1100px; margin:auto; }}
            .card {{ background:#0f172a; border:1px solid #1e293b; padding:25px; border-radius:10px; margin-bottom:25px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); }}
            video {{ width:100%; display:block; margin:auto; border:2px solid #38bdf8; border-radius:8px; }}
            #j-alert {{ display:none; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); background:rgba(239,68,68,0.95); color:white; padding:15px 30px; font-size:24px; font-weight:bold; border-radius:5px; border: 2px solid #fff; z-index:10; pointer-events:none; text-align:center; box-shadow: 0 0 25px rgba(239, 68, 68, 0.8); }}
            .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
            .box {{ background:#1e293b; padding:15px; border-radius:8px; text-align:center; border:1px dashed #475569; }}
            .box img {{ max-width:100%; border-radius:5px; border:2px solid #ef4444; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: #0f172a; border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #1e293b; }}
            th {{ background-color: #1e293b; color: #38bdf8; text-transform: uppercase; font-size: 0.85em; letter-spacing: 1px; }}
            .vector-box {{ width:100%; height:280px; background:#080d1a; display:flex; align-items:center; justify-content:center; border:2px solid #10b981; border-radius:5px; flex-direction:column; padding:10px; box-sizing:border-box; }}
            .vector-box svg {{ width: 160px; height: 160px; }}
            .wiki-btn {{ display:inline-block; padding:10px 15px; background:#10b981; color:#0f172a; border-radius:5px; text-decoration:none; font-weight:bold; margin-top:15px; font-size:0.9em; transition: 0.3s; }}
            .wiki-btn:hover {{ background:#34d399; box-shadow: 0 0 10px rgba(16, 185, 129, 0.5); }}
        </style></head><body>
        <h1 style="color:#00ffcc; letter-spacing:2px; text-transform:uppercase;">PlaySafe AI : Session Kinematic Report</h1>
        <p style="color:#94a3b8;">Source: <b>{src}</b> | AI Telemetry Confidence: <b>{acc:.1f}%</b></p>
        """

        if video_b64:
            html += f"""
            <div class="card" style="position:relative; text-align:center; background:#020617;">
                <h2 style="color:#10b981; margin-top:0;">▶️ JARVIS Interactive HUD Replay</h2>
                <p style="color:#cbd5e1; font-size:0.9em;">Replay features automatic fault-detection pauses and 0.3x slow-motion analysis.</p>
                <div style="position:relative; max-width: 900px; margin:auto;">
                    <div id="j-alert">SYSTEM HALT:<br><span id="atxt"></span></div>
                    <video id="vid" controls><source src="data:video/webm;base64,{video_b64}" type="video/webm"></video>
                </div>
            </div>
            """
        else:
            html += f"<div class='card'><p>Open <b>{os.path.basename(out_vid)}</b> directly in your media player to review HUD recording.</p></div>"

        html += f"""
        <script>
            const evs = {js}, vid = document.getElementById('vid'), al = document.getElementById('j-alert'), atxt = document.getElementById('atxt');
            let played = new Set();
            if (vid) {{
                vid.addEventListener('timeupdate', () => {{
                    evs.forEach((e, i) => {{
                        if (vid.currentTime >= e.time && vid.currentTime <= e.time + 0.4 && !played.has(i)) {{
                            vid.pause(); played.add(i); 
                            atxt.innerText = e.fault; al.style.display = 'block';
                            let m = new SpeechSynthesisUtterance(e.msg); m.rate = 1.1;
                            m.onend = () => {{ 
                                setTimeout(() => {{ 
                                    al.style.display = 'none'; 
                                    vid.playbackRate = 0.3; 
                                    vid.play(); 
                                    setTimeout(() => {{ vid.playbackRate = 1.0; }}, 3500); 
                                }}, 500); 
                            }};
                            window.speechSynthesis.speak(m);
                        }}
                    }});
                }});
                vid.addEventListener('seeked', () => {{ evs.forEach((e, i) => {{ if (vid.currentTime < e.time) played.delete(i); }}); }});
            }}
        </script>
        """

        if gemini_summary and "failed on Google" not in gemini_summary:
            html += f"<div class='card' style='border-left: 5px solid #a855f7;'><h2 style='color:#c084fc; margin-top:0;'>✨ Gemini AI Action Plan</h2><p style='line-height:1.6;'>{gemini_summary.replace(chr(10), '<br>')}</p></div>"

        html += "<h2>Biomechanical Deviations & Medical Analysis</h2>"
        if not self.snapshot_evidence:
             html += "<div class='card'><h3 style='color:#10b981;'>✅ Optimal Alignment Maintained. No biomechanical faults logged.</h3></div>"

        for f in self.snapshot_evidence:
            details = FAULT_DETAILS.get(f, {"meaning": "Unknown fault.", "fix": "Maintain neutral joint positioning.", "wiki": "https://en.wikipedia.org/wiki/Sports_injury"})
            
            # Use dynamic generator for correct SVG!
            svg_diagram = self.get_dynamic_svg(f)
            
            html += f"<div class='card'><h3 style='color:#ef4444; margin-top:0;'>⚠️ {f}</h3><p style='color:#cbd5e1;'><b>Kinematic Diagnosis:</b> {details['meaning']}</p>"
            for ts, b64 in self.snapshot_evidence[f]:
                time_str = f"{int(ts//60):02d}:{ts%60:05.2f}"
                html += f"""
                <div style="background:#3b82f6; color:white; padding:4px 12px; border-radius:12px; display:inline-block; font-size:0.85em; font-weight:bold; margin-bottom:12px;">T-MINUS {time_str}</div>
                <div class='grid'>
                    <div class='box'><h4 style='color:#ef4444; margin-top:0;'>Your Detected Form</h4><img src='data:image/jpeg;base64,{b64}'></div>
                    <div class='box'>
                        <h4 style='color:#10b981; margin-top:0;'>Structural Correction</h4>
                        <div class='vector-box'>{svg_diagram}<p style="color:#10b981; font-weight:bold; font-size:0.85em; margin:10px 0 0 0;">IDEAL BIOMECHANICAL VECTOR</p></div>
                        <p style="color: #f8fafc; font-size: 0.95em; margin-top: 15px; line-height: 1.5;"><b>Action Required:</b><br>{details['fix']}</p>
                        <a href="{details['wiki']}" target="_blank" class="wiki-btn">📚 Medical Reference (Wikipedia)</a>
                    </div>
                </div><br>
                """
            html += "</div>"
            
        html += "<h2>Range of Motion (ROM) Analysis</h2><table><tr><th>Joint Segment</th><th>Min Angle</th><th>Max Angle</th><th>Average Angle</th></tr>"
        for name, st in self.joint_stats.items():
            if st["count"] > 0: html += f"<tr><td>{name}</td><td>{int(st['min'])}&deg;</td><td>{int(st['max'])}&deg;</td><td>{int(st['sum']/st['count'])}&deg;</td></tr>"
            else: html += f"<tr><td>{name}</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>"
        html += "</table></body></html>"
        
        try:
            with open("PlaySafe_Interactive_Report.html", "w", encoding="utf-8") as file: 
                file.write(html)
            webbrowser.open(f"file://{os.path.abspath('PlaySafe_Interactive_Report.html')}")
        except Exception as e:
            print(f"Error launching web report: {e}")

class UIDrawer:
    @staticmethod
    def draw_tech_brackets(img, lms):
        h, w = img.shape[:2]
        xs = [int(lm.x * w) for lm in lms if is_landmark_reliable(lm)]
        ys = [int(lm.y * h) for lm in lms if is_landmark_reliable(lm)]
        if xs and ys:
            x1, y1 = max(0, min(xs) - 40), max(0, min(ys) - 40)
            x2, y2 = min(w, max(xs) + 40), min(h, max(ys) + 40)
            C = (255, 220, 0) 
            L = 30
            cv2.line(img, (x1, y1), (x1+L, y1), C, 2)
            cv2.line(img, (x1, y1), (x1, y1+L), C, 2)
            cv2.line(img, (x2, y1), (x2-L, y1), C, 2)
            cv2.line(img, (x2, y1), (x2, y1+L), C, 2)
            cv2.line(img, (x1, y2), (x1+L, y2), C, 2)
            cv2.line(img, (x1, y2), (x1, y2-L), C, 2)
            cv2.line(img, (x2, y2), (x2-L, y2), C, 2)
            cv2.line(img, (x2, y2), (x2, y2-L), C, 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.line(img, (cx-15, cy), (cx+15, cy), (0, 255, 100), 1)
            cv2.line(img, (cx, cy-15), (cx, cy+15), (0, 255, 100), 1)

    @staticmethod
    def snap(frame, lms, f):
        img = frame.copy()
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(frame.copy(), 0.4, img, 0.6, 0) 

        for idx in FAULT_LANDMARK_MAP.get(f, []):
            lm = get_landmark(lms, idx)
            if lm and is_landmark_reliable(lm):
                px, py = int(lm.x*w), int(lm.y*h)
                S, C = 22, (20,20,255)
                lines = [
                    ((px-S, py-S), (px-S+10, py-S)), ((px-S, py-S), (px-S, py-S+10)),
                    ((px+S, py-S), (px+S-10, py-S)), ((px+S, py-S), (px+S, py-S+10)),
                    ((px-S, py+S), (px-S+10, py+S)), ((px-S, py+S), (px-S, py+S-10)),
                    ((px+S, py+S), (px+S-10, py+S)), ((px+S, py+S), (px+S, py+S-10))
                ]
                for pt1, pt2 in lines: cv2.line(img, pt1, pt2, C, 2)
                cv2.circle(img, (px, py), 2, (240,240,255), -1)

        _, buf = cv2.imencode('.jpg', img)
        return base64.b64encode(buf).decode('utf-8')

    @staticmethod
    def draw(f, lms, faults):
        h, w = f.shape[:2]
        px = lambda lm: (int(lm.x * w), int(lm.y * h))
        if lms:
            UIDrawer.draw_tech_brackets(f, lms)
            for a, b in POSE_CONNECTIONS:
                la, lb = get_landmark(lms, a), get_landmark(lms, b)
                if is_landmark_reliable(la) and is_landmark_reliable(lb):
                    cv2.line(f, px(la), px(lb), (255, 255, 255), 2)
            for i in range(len(lms)):
                lm = get_landmark(lms, i)
                if is_landmark_reliable(lm): 
                    cv2.circle(f, px(lm), 4, (0, 255, 255), -1)
                    cv2.circle(f, px(lm), 8, (0, 200, 255), 1) 
        return f

    @staticmethod
    def draw_bar(val):
        if val is None: return "[..........]"
        fill = int((val / 180.0) * 10)
        fill = min(max(fill, 0), 10)
        return "[" + "■" * fill + "." * (10 - fill) + "]"

    @staticmethod
    def hud(f, ana, fps, t):
        h, w = f.shape[:2]
        c = np.full((h, w + 480, 3), (12, 12, 15), dtype=np.uint8) 
        c[:, :w] = f
        
        cv2.line(c, (w, 0), (w, h), (0, 255, 100), 2)
        cv2.putText(c, f"PLAYSAFE OS v11  |  FPS: {fps:.1f}  |  T: {t:.1f}s", (w + 25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 220, 0), 1)
        
        y_pos = 90
        cv2.putText(c, "LIVE SPATIAL TELEMETRY", (w + 25, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1)
        y_pos += 35
        
        for part in ["Shoulder", "Elbow", "Wrist", "Hip", "Knee"]:
            l_hist = ana.angle_history.get(f"Left {part}", [])
            r_hist = ana.angle_history.get(f"Right {part}", [])
            l_val = l_hist[-1] if l_hist and l_hist[-1] is not None else None
            r_val = r_hist[-1] if r_hist and r_hist[-1] is not None else None
            
            l_str = f"{int(l_val):03d}" if l_val is not None else "---"
            r_str = f"{int(r_val):03d}" if r_val is not None else "---"
            
            l_bar = UIDrawer.draw_bar(l_val)
            r_bar = UIDrawer.draw_bar(r_val)
            
            cv2.putText(c, f"{part.upper()}", (w + 25, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(c, f"L: {l_str} {l_bar}", (w + 25, y_pos + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 100), 1)
            cv2.putText(c, f"R: {r_str} {r_bar}", (w + 220, y_pos + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 200, 255), 1)
            y_pos += 45
            
        y_pos += 15
        cv2.putText(c, "ACTIVE FAULT STATUS", (w + 25, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1)
        y_pos += 30
        
        if ana.active_faults:
            for fault in ana.active_faults:
                cv2.rectangle(c, (w + 25, y_pos - 15), (w + 450, y_pos + 10), (0, 0, 150), -1)
                cv2.putText(c, f"[WARN] {fault.upper()}", (w + 35, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 35
        else:
            cv2.rectangle(c, (w + 25, y_pos - 15), (w + 450, y_pos + 10), (0, 100, 0), -1)
            cv2.putText(c, "[OK] KINEMATICS OPTIMAL", (w + 35, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        return c

def run(model, vid, cam):
    cap = cv2.VideoCapture(vid if vid else cam)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    out_vid = f"hud_output_{int(time.time())}.webm"
    width = int(cap.get(3)) if cap.get(3) > 0 else 640
    height = int(cap.get(4)) if cap.get(4) > 0 else 480
    
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(out_vid, fourcc, fps, (width + 480, height))
    
    opts = mp_vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=model), running_mode=mp_vision.RunningMode.VIDEO)
    ana = MovementAnalyzer()
    idx, dfps, st = 0, 0.0, time.time()
    pt = st

    voice_ai.speak("System initialized.", "BOOT", 0)

    with mp_vision.PoseLandmarker.create_from_options(opts) as lmkr:
        while True:
            ok, frm = cap.read()
            if not ok: break
            t = idx / fps
            idx += 1
            lms = None
            try:
                res = lmkr.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)), int(t * 1000))
                if res.pose_landmarks: lms = res.pose_landmarks[0]
            except: pass
            
            ana.update(lms, t)
            
            if lms and ana.pending_snapshots:
                for f in list(ana.pending_snapshots):
                    b64 = UIDrawer.snap(frm, lms, f)
                    if b64: 
                        ana.save_snap(f, b64, t)
                    ana.pending_snapshots.remove(f)
            
            uif = UIDrawer.draw(frm, lms, ana.active_faults)
            now = time.time()
            if now > pt: dfps = (dfps * 0.9) + (0.1 / (now - pt))
            pt = now
            
            c = UIDrawer.hud(uif, ana, dfps, t)
            out.write(c)
            cv2.imshow("PlaySafe OS", c)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\nProcessing complete. Fetching AI insights and compiling HTML report...")
    try:
        gemini_text = get_gemini_coach_insight(vid)
    except:
        gemini_text = ""
        
    ana.print_report(vid or "Live", out_vid, gemini_text)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--model", type=str, default="pose_landmarker_heavy.task")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()
    
    model_file = args.model if os.path.isfile(args.model) else "pose_landmarker_heavy.task"
    run(model_file, args.video, args.camera)