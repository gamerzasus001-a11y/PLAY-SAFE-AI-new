import cv2
import mediapipe as mp
import math
import time
from collections import deque


# ============================================================
#                 PLAYSAFE AI
#          SPORTS MOVEMENT SCREENING
# ============================================================
#
# MediaPipe Pose Landmarker
#
# Detects:
#   - Shoulders
#   - Elbows
#   - Wrists
#   - Hips
#   - Knees
#   - Ankles
#   - Feet
#
# Displays:
#   - Joint angles
#   - Visibility of body parts
#   - Movement information
#   - Caution node
#   - Caution branch
#
# IMPORTANT:
# This is a movement-screening prototype.
# It does NOT medically diagnose injuries.
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\PLAY SAFE AI\ai\pose_landmarker_full.task"

CAMERA_INDEX = 0

# Landmark must have at least this visibility
# to be considered reliable.
VISIBILITY_THRESHOLD = 0.60

# Number of previous values stored
HISTORY_LENGTH = 15

# Number of consecutive abnormal frames required
WARNING_FRAMES_REQUIRED = 5


# ============================================================
# MEDIAPIPE TASKS API
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculates angle ABC.

    a = first point
    b = middle/joint point
    c = third point

    Returns angle in degrees.
    """

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def distance(a, b):
    """
    Distance between two normalized points.
    """

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def get_point(landmark):
    """
    Convert MediaPipe landmark into x,y.
    """

    return (
        landmark.x,
        landmark.y
    )


def is_visible(landmark, threshold=VISIBILITY_THRESHOLD):
    """
    Checks whether MediaPipe considers the landmark visible.
    """

    try:
        return landmark.visibility >= threshold
    except:
        return False


def all_visible(*landmarks):
    """
    Returns True only when every landmark is visible.
    """

    return all(
        is_visible(lm)
        for lm in landmarks
    )


def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.50,
    thickness=1
):
    """
    Draw standard white text.
    """

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_node(
    frame,
    landmark,
    width,
    height,
    highlighted=False
):
    """
    Draw a joint/node.
    """

    if not is_visible(landmark):
        return

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    if highlighted:

        # Outer circle
        cv2.circle(
            frame,
            (x, y),
            13,
            (0, 0, 255),
            3
        )

        # Inner circle
        cv2.circle(
            frame,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

    else:

        cv2.circle(
            frame,
            (x, y),
            5,
            (255, 255, 255),
            -1
        )


def draw_branch(
    frame,
    lm1,
    lm2,
    width,
    height,
    highlighted=False
):
    """
    Draw body branch.
    """

    if not all_visible(lm1, lm2):
        return

    x1 = int(lm1.x * width)
    y1 = int(lm1.y * height)

    x2 = int(lm2.x * width)
    y2 = int(lm2.y * height)

    if highlighted:

        # Red caution branch
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            6
        )

    else:

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


def angle_text(angle):
    """
    Convert angle to display text.
    """

    if angle is None:
        return "NOT SHOWN IN VIDEO"

    return f"{angle:.0f} deg"


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("")
    print("ERROR: Could not open camera.")
    print("Check that your webcam is available.")
    exit()


print("")
print("============================================")
print("             PLAYSAFE AI")
print("        MOVEMENT SCREENING SYSTEM")
print("============================================")
print("")
print("Camera started.")
print("Press Q to quit.")
print("")


# ============================================================
# HISTORY
# ============================================================

left_knee_history = deque(
    maxlen=HISTORY_LENGTH
)

right_knee_history = deque(
    maxlen=HISTORY_LENGTH
)

left_elbow_history = deque(
    maxlen=HISTORY_LENGTH
)

right_elbow_history = deque(
    maxlen=HISTORY_LENGTH
)

left_ankle_history = deque(
    maxlen=HISTORY_LENGTH
)

right_ankle_history = deque(
    maxlen=HISTORY_LENGTH
)

left_wrist_history = deque(
    maxlen=HISTORY_LENGTH
)

right_wrist_history = deque(
    maxlen=HISTORY_LENGTH
)


# ============================================================
# WARNING COUNTERS
# ============================================================

left_knee_warning_count = 0
right_knee_warning_count = 0

left_elbow_warning_count = 0
right_elbow_warning_count = 0

left_ankle_warning_count = 0
right_ankle_warning_count = 0

fall_warning_count = 0


# ============================================================
# PREVIOUS POSITIONS
# ============================================================

previous_left_wrist = None
previous_right_wrist = None

previous_time = time.time()


# ============================================================
# START TIME
# ============================================================

start_time = time.time()


# ============================================================
# POSE LANDMARKER
# ============================================================

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        success, frame = cap.read()

        if not success:

            print("ERROR: Could not read camera frame.")
            break


        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )


        height, width, _ = frame.shape


        # ====================================================
        # CONVERT TO RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # ====================================================
        # DETECT POSE
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        status = "NO PERSON DETECTED"

        caution_node = "NONE"

        caution_branch = "NONE"

        caution_message = ""


        # ====================================================
        # PERSON DETECTED
        # ====================================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]


            # =================================================
            # BODY LANDMARKS
            # =================================================

            nose = landmarks[0]

            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            left_elbow = landmarks[13]
            right_elbow = landmarks[14]

            left_wrist = landmarks[15]
            right_wrist = landmarks[16]

            left_hip = landmarks[23]
            right_hip = landmarks[24]

            left_knee = landmarks[25]
            right_knee = landmarks[26]

            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            left_foot = landmarks[31]
            right_foot = landmarks[32]


            # =================================================
            # DETERMINE VISIBILITY
            # =================================================

            left_arm_visible = all_visible(
                left_shoulder,
                left_elbow,
                left_wrist
            )

            right_arm_visible = all_visible(
                right_shoulder,
                right_elbow,
                right_wrist
            )


            # -------------------------------------------------
            # IMPORTANT KNEE BUG FIX
            # -------------------------------------------------
            #
            # Knee is ONLY analyzed if:
            #
            # HIP + KNEE + ANKLE
            #
            # are all visible.
            #
            # Therefore upper-body-only video will not
            # randomly trigger a knee warning.
            # -------------------------------------------------

            left_leg_visible = all_visible(
                left_hip,
                left_knee,
                left_ankle
            )

            right_leg_visible = all_visible(
                right_hip,
                right_knee,
                right_ankle
            )


            left_ankle_angle_visible = all_visible(
                left_knee,
                left_ankle,
                left_foot
            )

            right_ankle_angle_visible = all_visible(
                right_knee,
                right_ankle,
                right_foot
            )


            # =================================================
            # DRAW BODY BRANCHES
            # =================================================

            # Shoulders
            draw_branch(
                frame,
                left_shoulder,
                right_shoulder,
                width,
                height
            )

            # Left arm
            draw_branch(
                frame,
                left_shoulder,
                left_elbow,
                width,
                height
            )

            draw_branch(
                frame,
                left_elbow,
                left_wrist,
                width,
                height
            )

            # Right arm
            draw_branch(
                frame,
                right_shoulder,
                right_elbow,
                width,
                height
            )

            draw_branch(
                frame,
                right_elbow,
                right_wrist,
                width,
                height
            )

            # Torso
            draw_branch(
                frame,
                left_shoulder,
                left_hip,
                width,
                height
            )

            draw_branch(
                frame,
                right_shoulder,
                right_hip,
                width,
                height
            )

            draw_branch(
                frame,
                left_hip,
                right_hip,
                width,
                height
            )

            # Left leg
            draw_branch(
                frame,
                left_hip,
                left_knee,
                width,
                height
            )

            draw_branch(
                frame,
                left_knee,
                left_ankle,
                width,
                height
            )

            # Right leg
            draw_branch(
                frame,
                right_hip,
                right_knee,
                width,
                height
            )

            draw_branch(
                frame,
                right_knee,
                right_ankle,
                width,
                height
            )


            # Feet
            draw_branch(
                frame,
                left_ankle,
                left_foot,
                width,
                height
            )

            draw_branch(
                frame,
                right_ankle,
                right_foot,
                width,
                height
            )


            # =================================================
            # KNEE ANGLES
            # =================================================

            left_knee_angle = None
            right_knee_angle = None


            if left_leg_visible:

                left_knee_angle = calculate_angle(
                    get_point(left_hip),
                    get_point(left_knee),
                    get_point(left_ankle)
                )

                left_knee_history.append(
                    left_knee_angle
                )

            else:

                left_knee_history.clear()

                left_knee_warning_count = 0


            if right_leg_visible:

                right_knee_angle = calculate_angle(
                    get_point(right_hip),
                    get_point(right_knee),
                    get_point(right_ankle)
                )

                right_knee_history.append(
                    right_knee_angle
                )

            else:

                right_knee_history.clear()

                right_knee_warning_count = 0


            # =================================================
            # ELBOW ANGLES
            # =================================================

            left_elbow_angle = None
            right_elbow_angle = None


            if left_arm_visible:

                left_elbow_angle = calculate_angle(
                    get_point(left_shoulder),
                    get_point(left_elbow),
                    get_point(left_wrist)
                )

                left_elbow_history.append(
                    left_elbow_angle
                )

            else:

                left_elbow_history.clear()


            if right_arm_visible:

                right_elbow_angle = calculate_angle(
                    get_point(right_shoulder),
                    get_point(right_elbow),
                    get_point(right_wrist)
                )

                right_elbow_history.append(
                    right_elbow_angle
                )

            else:

                right_elbow_history.clear()


            # =================================================
            # ANKLE ANGLES
            # =================================================

            left_ankle_angle = None
            right_ankle_angle = None


            if left_ankle_angle_visible:

                left_ankle_angle = calculate_angle(
                    get_point(left_knee),
                    get_point(left_ankle),
                    get_point(left_foot)
                )

                left_ankle_history.append(
                    left_ankle_angle
                )

            else:

                left_ankle_history.clear()


            if right_ankle_angle_visible:

                right_ankle_angle = calculate_angle(
                    get_point(right_knee),
                    get_point(right_ankle),
                    get_point(right_foot)
                )

                right_ankle_history.append(
                    right_ankle_angle
                )

            else:

                right_ankle_history.clear()


            # =================================================
            # WRIST MOVEMENT
            # =================================================

            current_time = time.time()

            dt = current_time - previous_time

            if dt <= 0:
                dt = 0.001


            left_wrist_speed = 0
            right_wrist_speed = 0


            # LEFT WRIST
            if is_visible(left_wrist):

                current_left_wrist = get_point(
                    left_wrist
                )

                left_wrist_history.append(
                    current_left_wrist
                )


                if previous_left_wrist is not None:

                    left_wrist_speed = (
                        distance(
                            current_left_wrist,
                            previous_left_wrist
                        )
                        /
                        dt
                    )


                previous_left_wrist = (
                    current_left_wrist
                )

            else:

                left_wrist_history.clear()

                previous_left_wrist = None


            # RIGHT WRIST
            if is_visible(right_wrist):

                current_right_wrist = get_point(
                    right_wrist
                )

                right_wrist_history.append(
                    current_right_wrist
                )


                if previous_right_wrist is not None:

                    right_wrist_speed = (
                        distance(
                            current_right_wrist,
                            previous_right_wrist
                        )
                        /
                        dt
                    )


                previous_right_wrist = (
                    current_right_wrist
                )

            else:

                right_wrist_history.clear()

                previous_right_wrist = None


            previous_time = current_time


            # =================================================
            # KNEE CAUTION ANALYSIS
            # =================================================

            left_knee_abnormal = False
            right_knee_abnormal = False


            # Need multiple frames
            if (
                left_leg_visible
                and
                len(left_knee_history) >= 5
            ):

                recent_values = list(
                    left_knee_history
                )[-5:]


                # Extreme knee flexion
                if all(
                    value < 55
                    for value in recent_values
                ):

                    left_knee_abnormal = True


            if (
                right_leg_visible
                and
                len(right_knee_history) >= 5
            ):

                recent_values = list(
                    right_knee_history
                )[-5:]


                if all(
                    value < 55
                    for value in recent_values
                ):

                    right_knee_abnormal = True


            # =================================================
            # KNEE COUNTERS
            # =================================================

            if left_knee_abnormal:

                left_knee_warning_count += 1

            else:

                left_knee_warning_count = 0


            if right_knee_abnormal:

                right_knee_warning_count += 1

            else:

                right_knee_warning_count = 0


            # =================================================
            # ELBOW CAUTION
            # =================================================

            left_elbow_abnormal = False
            right_elbow_abnormal = False


            if (
                left_elbow_angle is not None
                and
                len(left_elbow_history) >= 5
            ):

                recent = list(
                    left_elbow_history
                )[-5:]


                if all(
                    value < 35
                    for value in recent
                ):

                    left_elbow_abnormal = True


            if (
                right_elbow_angle is not None
                and
                len(right_elbow_history) >= 5
            ):

                recent = list(
                    right_elbow_history
                )[-5:]


                if all(
                    value < 35
                    for value in recent
                ):

                    right_elbow_abnormal = True


            if left_elbow_abnormal:

                left_elbow_warning_count += 1

            else:

                left_elbow_warning_count = 0


            if right_elbow_abnormal:

                right_elbow_warning_count += 1

            else:

                right_elbow_warning_count = 0


            # =================================================
            # ANKLE CAUTION
            # =================================================

            left_ankle_abnormal = False
            right_ankle_abnormal = False


            if left_ankle_angle is not None:

                if (
                    left_ankle_angle < 55
                    or
                    left_ankle_angle > 160
                ):

                    left_ankle_abnormal = True


            if right_ankle_angle is not None:

                if (
                    right_ankle_angle < 55
                    or
                    right_ankle_angle > 160
                ):

                    right_ankle_abnormal = True


            if left_ankle_abnormal:

                left_ankle_warning_count += 1

            else:

                left_ankle_warning_count = 0


            if right_ankle_abnormal:

                right_ankle_warning_count += 1

            else:

                right_ankle_warning_count = 0


            # =================================================
            # FALL / UNUSUAL POSTURE
            # =================================================

            fall_detected = False


            if all_visible(
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            ):

                shoulder_x = (
                    left_shoulder.x +
                    right_shoulder.x
                ) / 2


                shoulder_y = (
                    left_shoulder.y +
                    right_shoulder.y
                ) / 2


                hip_x = (
                    left_hip.x +
                    right_hip.x
                ) / 2


                hip_y = (
                    left_hip.y +
                    right_hip.y
                ) / 2


                torso_width = abs(
                    right_shoulder.x -
                    left_shoulder.x
                )


                torso_height = abs(
                    hip_y -
                    shoulder_y
                )


                if (
                    torso_width > 0
                    and
                    torso_height
                    <
                    torso_width * 0.55
                ):

                    fall_detected = True


            if fall_detected:

                fall_warning_count += 1

            else:

                fall_warning_count = 0


            # =================================================
            # SELECT MOST IMPORTANT CAUTION
            # =================================================

            # Knee gets priority
            if (
                left_knee_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "LEFT KNEE (25)"

                caution_branch = (
                    "LEFT HIP -> LEFT KNEE -> LEFT ANKLE"
                )

                caution_message = (
                    "CAUTION: LEFT KNEE MOVEMENT"
                )


            elif (
                right_knee_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "RIGHT KNEE (26)"

                caution_branch = (
                    "RIGHT HIP -> RIGHT KNEE -> RIGHT ANKLE"
                )

                caution_message = (
                    "CAUTION: RIGHT KNEE MOVEMENT"
                )


            elif (
                left_elbow_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "LEFT ELBOW (13)"

                caution_branch = (
                    "LEFT SHOULDER -> LEFT ELBOW -> LEFT WRIST"
                )

                caution_message = (
                    "CAUTION: LEFT ELBOW MOVEMENT"
                )


            elif (
                right_elbow_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "RIGHT ELBOW (14)"

                caution_branch = (
                    "RIGHT SHOULDER -> RIGHT ELBOW -> RIGHT WRIST"
                )

                caution_message = (
                    "CAUTION: RIGHT ELBOW MOVEMENT"
                )


            elif (
                left_ankle_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "LEFT ANKLE (27)"

                caution_branch = (
                    "LEFT KNEE -> LEFT ANKLE -> LEFT FOOT"
                )

                caution_message = (
                    "CAUTION: LEFT ANKLE MOVEMENT"
                )


            elif (
                right_ankle_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "RIGHT ANKLE (28)"

                caution_branch = (
                    "RIGHT KNEE -> RIGHT ANKLE -> RIGHT FOOT"
                )

                caution_message = (
                    "CAUTION: RIGHT ANKLE MOVEMENT"
                )


            elif (
                fall_warning_count
                >= WARNING_FRAMES_REQUIRED
            ):

                caution_node = "FULL BODY"

                caution_branch = (
                    "SHOULDER -> HIP"
                )

                caution_message = (
                    "CAUTION: UNUSUAL POSTURE / POSSIBLE FALL"
                )


            # =================================================
            # HIGHLIGHT CAUTION BRANCH
            # =================================================

            if "LEFT KNEE" in caution_node:

                draw_branch(
                    frame,
                    left_hip,
                    left_knee,
                    width,
                    height,
                    True
                )

                draw_branch(
                    frame,
                    left_knee,
                    left_ankle,
                    width,
                    height,
                    True
                )


            elif "RIGHT KNEE" in caution_node:

                draw_branch(
                    frame,
                    right_hip,
                    right_knee,
                    width,
                    height,
                    True
                )

                draw_branch(
                    frame,
                    right_knee,
                    right_ankle,
                    width,
                    height,
                    True
                )


            elif "LEFT ELBOW" in caution_node:

                draw_branch(
                    frame,
                    left_shoulder,
                    left_elbow,
                    width,
                    height,
                    True
                )

                draw_branch(
                    frame,
                    left_elbow,
                    left_wrist,
                    width,
                    height,
                    True
                )


            elif "RIGHT ELBOW" in caution_node:

                draw_branch(
                    frame,
                    right_shoulder,
                    right_elbow,
                    width,
                    height,
                    True
                )

                draw_branch(
                    frame,
                    right_elbow,
                    right_wrist,
                    width,
                    height,
                    True
                )


            elif "LEFT ANKLE" in caution_node:

                draw_branch(
                    frame,
                    left_knee,
                    left_ankle,
                    width,
                    height,
                    True
                )

                draw_branch(
                    frame,
                    left_ankle,
                    left_foot,
                    width,
                    height,
                    True
                )


            elif "RIGHT ANKLE" in caution_node:

                draw_branch(
                    frame,
                    right_knee,
                    right_ankle,
                    width,
                    height,
                    True
                )

                draw_branch(
                    frame,
                    right_ankle,
                    right_foot,
                    width,
                    height,
                    True
                )


            # =================================================
            # DRAW NODES
            # =================================================

            # Normal nodes first

            draw_node(
                frame,
                left_shoulder,
                width,
                height
            )

            draw_node(
                frame,
                right_shoulder,
                width,
                height
            )

            draw_node(
                frame,
                left_elbow,
                width,
                height
            )

            draw_node(
                frame,
                right_elbow,
                width,
                height
            )

            draw_node(
                frame,
                left_wrist,
                width,
                height
            )

            draw_node(
                frame,
                right_wrist,
                width,
                height
            )

            draw_node(
                frame,
                left_hip,
                width,
                height
            )

            draw_node(
                frame,
                right_hip,
                width,
                height
            )

            draw_node(
                frame,
                left_knee,
                width,
                height
            )

            draw_node(
                frame,
                right_knee,
                width,
                height
            )

            draw_node(
                frame,
                left_ankle,
                width,
                height
            )

            draw_node(
                frame,
                right_ankle,
                width,
                height
            )


            # Highlight caution node

            if "LEFT KNEE" in caution_node:

                draw_node(
                    frame,
                    left_knee,
                    width,
                    height,
                    True
                )


            elif "RIGHT KNEE" in caution_node:

                draw_node(
                    frame,
                    right_knee,
                    width,
                    height,
                    True
                )


            elif "LEFT ELBOW" in caution_node:

                draw_node(
                    frame,
                    left_elbow,
                    width,
                    height,
                    True
                )


            elif "RIGHT ELBOW" in caution_node:

                draw_node(
                    frame,
                    right_elbow,
                    width,
                    height,
                    True
                )


            elif "LEFT ANKLE" in caution_node:

                draw_node(
                    frame,
                    left_ankle,
                    width,
                    height,
                    True
                )


            elif "RIGHT ANKLE" in caution_node:

                draw_node(
                    frame,
                    right_ankle,
                    width,
                    height,
                    True
                )


            # =================================================
            # STATUS
            # =================================================

            if caution_message:

                status = "CAUTION"

            else:

                status = "MOVEMENT OK"


            # =================================================
            # INFORMATION PANEL
            # =================================================

            panel_width = 385

            panel_height = 395


            overlay = frame.copy()


            cv2.rectangle(
                overlay,
                (10, 85),
                (
                    panel_width,
                    85 + panel_height
                ),
                (20, 20, 20),
                -1
            )


            # Slight transparency
            frame = cv2.addWeighted(
                overlay,
                0.80,
                frame,
                0.20,
                0
            )


            # =================================================
            # PANEL TITLE
            # =================================================

            draw_text(
                frame,
                "BODY ANALYSIS",
                25,
                112,
                0.70,
                2
            )


            # =================================================
            # KNEES
            # =================================================

            draw_text(
                frame,
                "LEFT KNEE:",
                25,
                145,
                0.48,
                1
            )


            if left_knee_angle is None:

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    145,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    f"{left_knee_angle:.0f} deg",
                    145,
                    145,
                    0.50,
                    1
                )


            draw_text(
                frame,
                "RIGHT KNEE:",
                25,
                170,
                0.48,
                1
            )


            if right_knee_angle is None:

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    170,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    f"{right_knee_angle:.0f} deg",
                    145,
                    170,
                    0.50,
                    1
                )


            # =================================================
            # ELBOWS
            # =================================================

            draw_text(
                frame,
                "LEFT ELBOW:",
                25,
                200,
                0.48,
                1
            )


            if left_elbow_angle is None:

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    200,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    f"{left_elbow_angle:.0f} deg",
                    145,
                    200,
                    0.50,
                    1
                )


            draw_text(
                frame,
                "RIGHT ELBOW:",
                25,
                225,
                0.48,
                1
            )


            if right_elbow_angle is None:

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    225,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    f"{right_elbow_angle:.0f} deg",
                    145,
                    225,
                    0.50,
                    1
                )


            # =================================================
            # ANKLES
            # =================================================

            draw_text(
                frame,
                "LEFT ANKLE:",
                25,
                255,
                0.48,
                1
            )


            if left_ankle_angle is None:

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    255,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    f"{left_ankle_angle:.0f} deg",
                    145,
                    255,
                    0.50,
                    1
                )


            draw_text(
                frame,
                "RIGHT ANKLE:",
                25,
                280,
                0.48,
                1
            )


            if right_ankle_angle is None:

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    280,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    f"{right_ankle_angle:.0f} deg",
                    145,
                    280,
                    0.50,
                    1
                )


            # =================================================
            # WRISTS
            # =================================================

            draw_text(
                frame,
                "LEFT WRIST:",
                25,
                310,
                0.48,
                1
            )


            if not is_visible(left_wrist):

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    310,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    "TRACKING",
                    145,
                    310,
                    0.45,
                    1
                )


            draw_text(
                frame,
                "RIGHT WRIST:",
                25,
                335,
                0.48,
                1
            )


            if not is_visible(right_wrist):

                draw_text(
                    frame,
                    "NOT SHOWN IN VIDEO",
                    145,
                    335,
                    0.42,
                    1
                )

            else:

                draw_text(
                    frame,
                    "TRACKING",
                    145,
                    335,
                    0.45,
                    1
                )


            # =================================================
            # STATUS
            # =================================================

            draw_text(
                frame,
                "STATUS:",
                25,
                365,
                0.50,
                2
            )


            draw_text(
                frame,
                status,
                110,
                365,
                0.50,
                2
            )


        # ====================================================
        # NO PERSON
        # ====================================================

        else:

            status = "NO PERSON DETECTED"

            caution_node = "NONE"

            caution_branch = "NONE"

            caution_message = ""


            # Clear histories
            left_knee_history.clear()
            right_knee_history.clear()

            left_elbow_history.clear()
            right_elbow_history.clear()

            left_ankle_history.clear()
            right_ankle_history.clear()

            left_wrist_history.clear()
            right_wrist_history.clear()


            # Reset counters
            left_knee_warning_count = 0
            right_knee_warning_count = 0

            left_elbow_warning_count = 0
            right_elbow_warning_count = 0

            left_ankle_warning_count = 0
            right_ankle_warning_count = 0

            fall_warning_count = 0


            previous_left_wrist = None
            previous_right_wrist = None


        # ====================================================
        # TOP BAR
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 70),
            (25, 25, 25),
            -1
        )


        draw_text(
            frame,
            "PLAYSAFE AI",
            20,
            45,
            1.0,
            2
        )


        draw_text(
            frame,
            status,
            width - 300,
            45,
            0.60,
            2
        )


        # ====================================================
        # CAUTION INFORMATION
        # ====================================================

        if caution_message:

            caution_height = 130

            caution_top = (
                height -
                caution_height -
                15
            )


            cv2.rectangle(
                frame,
                (15, caution_top),
                (
                    width - 15,
                    height - 15
                ),
                (35, 35, 35),
                -1
            )


            draw_text(
                frame,
                caution_message,
                30,
                caution_top + 30,
                0.60,
                2
            )


            draw_text(
                frame,
                "CAUTION NODE: "
                + caution_node,
                30,
                caution_top + 60,
                0.48,
                1
            )


            draw_text(
                frame,
                "BRANCH: "
                + caution_branch,
                30,
                caution_top + 88,
                0.43,
                1
            )


            draw_text(
                frame,
                "Movement screening only - "
                "not a medical diagnosis",
                30,
                caution_top + 113,
                0.38,
                1
            )


        # ====================================================
        # BOTTOM INSTRUCTION
        # ====================================================

        draw_text(
            frame,
            "Press Q to quit",
            20,
            height - 10,
            0.42,
            1
        )


        # ====================================================
        # SHOW FRAME
        # ====================================================

        cv2.imshow(
            "PlaySafe AI - Movement Analysis",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print("")
print("============================================")
print("        PlaySafe AI stopped.")
print("============================================")