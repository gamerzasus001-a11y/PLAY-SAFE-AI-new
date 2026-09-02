import cv2
import mediapipe as mp
import math
import os
import time


# ============================================================
# PLAYSAFE AI
# FULL BODY MOVEMENT SCREENING
#
# Detects/screens:
#   - Knee
#   - Ankle
#   - Elbow
#   - Wrist
#   - Shoulder
#   - Hip
#   - Torso
#   - Fall / sudden movement
#
# IMPORTANT:
# This is a movement-screening system.
# It does NOT diagnose injuries.
# ============================================================


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "pose_landmarker_full.task"
)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: pose_landmarker_full.task was not found.")
    print("Expected location:")
    print(MODEL_PATH)
    print("\nPut pose_landmarker_full.task inside the ai folder.")
    exit()


# ============================================================
# 2. MEDIAPIPE TASKS SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


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
# 3. MEDIAPIPE LANDMARK NUMBERS
# ============================================================

NOSE = 0

LEFT_EYE = 2
RIGHT_EYE = 5

LEFT_EAR = 7
RIGHT_EAR = 8

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

LEFT_HEEL = 29
RIGHT_HEEL = 30

LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ============================================================
# 4. SETTINGS
# ============================================================

VISIBILITY_THRESHOLD = 0.50

# Angle thresholds are screening thresholds.
# They are NOT medical injury thresholds.

KNEE_CAUTION_ANGLE = 70

ELBOW_CAUTION_ANGLE = 25

ANKLE_LOW_ANGLE = 45
ANKLE_HIGH_ANGLE = 140

SHOULDER_LOW_ANGLE = 20

HIP_LOW_ANGLE = 35

TORSO_LEAN_THRESHOLD = 35

# Fall detection sensitivity.
FALL_VERTICAL_CHANGE = 0.18
FALL_HORIZONTAL_BODY_ANGLE = 55


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def landmark_visible(
    landmark,
    threshold=VISIBILITY_THRESHOLD
):
    """
    Checks whether MediaPipe considers a landmark
    reliable enough to use.
    """

    if landmark is None:
        return False

    try:
        return landmark.visibility >= threshold

    except Exception:
        return False


def get_landmark(
    pose,
    index
):
    """
    Safely retrieves a landmark.
    """

    if pose is None:
        return None

    if index < 0 or index >= len(pose):
        return None

    return pose[index]


def calculate_angle(
    a,
    b,
    c
):
    """
    Calculates angle ABC.
    """

    if a is None or b is None or c is None:
        return None

    angle = math.degrees(
        math.atan2(
            c.y - b.y,
            c.x - b.x
        )
        -
        math.atan2(
            a.y - b.y,
            a.x - b.x
        )
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def point_on_screen(
    landmark,
    width,
    height
):
    """
    Converts MediaPipe normalized coordinates
    to camera coordinates.
    """

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    return x, y


def midpoint(
    a,
    b
):
    """
    Returns midpoint of two landmarks.
    """

    if a is None or b is None:
        return None

    class Point:
        pass

    p = Point()

    p.x = (a.x + b.x) / 2
    p.y = (a.y + b.y) / 2

    return p


def put_text(
    frame,
    text,
    x,
    y,
    color=(255, 255, 255),
    size=0.60
):
    """
    Draws readable text.
    """

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# 6. DRAWING FUNCTIONS
# ============================================================

def draw_branch(
    frame,
    pose,
    point1_index,
    point2_index,
    width,
    height,
    concerning=False
):
    """
    Draws a body branch.

    GREEN = normal/reliably tracked
    RED   = potentially concerning movement
    """

    p1 = get_landmark(
        pose,
        point1_index
    )

    p2 = get_landmark(
        pose,
        point2_index
    )

    if p1 is None or p2 is None:
        return

    if not landmark_visible(p1):
        return

    if not landmark_visible(p2):
        return

    point1 = point_on_screen(
        p1,
        width,
        height
    )

    point2 = point_on_screen(
        p2,
        width,
        height
    )

    if concerning:

        color = (0, 0, 255)
        thickness = 7

    else:

        color = (0, 255, 0)
        thickness = 4

    cv2.line(
        frame,
        point1,
        point2,
        color,
        thickness
    )


def draw_joint(
    frame,
    pose,
    index,
    width,
    height,
    concerning=False
):
    """
    Draws an individual joint.
    """

    landmark = get_landmark(
        pose,
        index
    )

    if landmark is None:
        return

    if not landmark_visible(landmark):
        return

    x, y = point_on_screen(
        landmark,
        width,
        height
    )

    if concerning:

        color = (0, 0, 255)
        radius = 9

    else:

        color = (0, 255, 0)
        radius = 6

    cv2.circle(
        frame,
        (x, y),
        radius,
        color,
        -1
    )


# ============================================================
# 7. DRAW COMPLETE BODY
# ============================================================

def draw_body(
    frame,
    pose,
    width,
    height,
    warnings
):
    """
    Draws the entire skeleton.

    Only the relevant concerning branch becomes red.
    """

    connections = [

        # --------------------------
        # LEFT ARM
        # --------------------------

        (
            LEFT_SHOULDER,
            LEFT_ELBOW,
            warnings["left_elbow"]
        ),

        (
            LEFT_ELBOW,
            LEFT_WRIST,
            warnings["left_elbow"]
        ),

        # --------------------------
        # RIGHT ARM
        # --------------------------

        (
            RIGHT_SHOULDER,
            RIGHT_ELBOW,
            warnings["right_elbow"]
        ),

        (
            RIGHT_ELBOW,
            RIGHT_WRIST,
            warnings["right_elbow"]
        ),

        # --------------------------
        # SHOULDERS
        # --------------------------

        (
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            warnings["shoulder"]
        ),

        # --------------------------
        # LEFT TORSO
        # --------------------------

        (
            LEFT_SHOULDER,
            LEFT_HIP,
            warnings["torso"]
        ),

        # --------------------------
        # RIGHT TORSO
        # --------------------------

        (
            RIGHT_SHOULDER,
            RIGHT_HIP,
            warnings["torso"]
        ),

        # --------------------------
        # HIPS
        # --------------------------

        (
            LEFT_HIP,
            RIGHT_HIP,
            warnings["hip"]
        ),

        # --------------------------
        # LEFT LEG
        # --------------------------

        (
            LEFT_HIP,
            LEFT_KNEE,
            warnings["left_knee"]
        ),

        (
            LEFT_KNEE,
            LEFT_ANKLE,
            warnings["left_knee"]
        ),

        (
            LEFT_ANKLE,
            LEFT_HEEL,
            warnings["left_ankle"]
        ),

        (
            LEFT_HEEL,
            LEFT_FOOT_INDEX,
            warnings["left_ankle"]
        ),

        # --------------------------
        # RIGHT LEG
        # --------------------------

        (
            RIGHT_HIP,
            RIGHT_KNEE,
            warnings["right_knee"]
        ),

        (
            RIGHT_KNEE,
            RIGHT_ANKLE,
            warnings["right_knee"]
        ),

        (
            RIGHT_ANKLE,
            RIGHT_HEEL,
            warnings["right_ankle"]
        ),

        (
            RIGHT_HEEL,
            RIGHT_FOOT_INDEX,
            warnings["right_ankle"]
        )
    ]


    # Draw branches

    for p1, p2, warning in connections:

        draw_branch(
            frame,
            pose,
            p1,
            p2,
            width,
            height,
            warning
        )


    # ========================================================
    # DRAW JOINTS
    # ========================================================

    joints = [

        # Left side

        (
            LEFT_SHOULDER,
            warnings["shoulder"]
        ),

        (
            LEFT_ELBOW,
            warnings["left_elbow"]
        ),

        (
            LEFT_WRIST,
            warnings["left_wrist"]
        ),

        (
            LEFT_HIP,
            warnings["hip"]
        ),

        (
            LEFT_KNEE,
            warnings["left_knee"]
        ),

        (
            LEFT_ANKLE,
            warnings["left_ankle"]
        ),

        # Right side

        (
            RIGHT_SHOULDER,
            warnings["shoulder"]
        ),

        (
            RIGHT_ELBOW,
            warnings["right_elbow"]
        ),

        (
            RIGHT_WRIST,
            warnings["right_wrist"]
        ),

        (
            RIGHT_HIP,
            warnings["hip"]
        ),

        (
            RIGHT_KNEE,
            warnings["right_knee"]
        ),

        (
            RIGHT_ANKLE,
            warnings["right_ankle"]
        )
    ]


    for index, warning in joints:

        draw_joint(
            frame,
            pose,
            index,
            width,
            height,
            warning
        )


# ============================================================
# 8. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open camera.")
    print("Check if another program is using the webcam.")

    exit()


# Request 1280x720.

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# 9. LARGE WINDOW
# ============================================================

WINDOW_NAME = "PlaySafe AI"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)


# ============================================================
# 10. FALL DETECTION MEMORY
# ============================================================

previous_body_y = None

previous_time = time.time()

fall_cooldown = 0

FALL_COOLDOWN_FRAMES = 60


# ============================================================
# 11. MEDIAPIPE
# ============================================================

with PoseLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        success, frame = cap.read()

        if not success:

            print("Could not read camera frame.")

            break


        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        height, width, _ = frame.shape


        # ====================================================
        # RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        frame_timestamp_ms += 33


        # ====================================================
        # POSE DETECTION
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )


        # ====================================================
        # DEFAULT STATUS
        # ====================================================

        body_status = "BODY: NOT SHOWN"

        left_knee_status = "LEFT KNEE: NOT SHOWN"
        right_knee_status = "RIGHT KNEE: NOT SHOWN"

        left_ankle_status = "LEFT ANKLE: NOT SHOWN"
        right_ankle_status = "RIGHT ANKLE: NOT SHOWN"

        left_elbow_status = "LEFT ELBOW: NOT SHOWN"
        right_elbow_status = "RIGHT ELBOW: NOT SHOWN"

        left_wrist_status = "LEFT WRIST: NOT SHOWN"
        right_wrist_status = "RIGHT WRIST: NOT SHOWN"

        shoulder_status = "SHOULDER: NOT SHOWN"

        hip_status = "HIP: NOT SHOWN"

        torso_status = "TORSO: NOT SHOWN"

        fall_status = ""

        # ====================================================
        # WARNING FLAGS
        # ====================================================

        warnings = {

            "left_knee": False,
            "right_knee": False,

            "left_ankle": False,
            "right_ankle": False,

            "left_elbow": False,
            "right_elbow": False,

            "left_wrist": False,
            "right_wrist": False,

            "shoulder": False,

            "hip": False,

            "torso": False,

            "fall": False
        }


        # ====================================================
        # PERSON FOUND
        # ====================================================

        if result.pose_landmarks:

            pose = result.pose_landmarks[0]

            body_status = "BODY: DETECTED"


            # =================================================
            # GET LANDMARKS
            # =================================================

            nose = get_landmark(
                pose,
                NOSE
            )

            left_shoulder = get_landmark(
                pose,
                LEFT_SHOULDER
            )

            right_shoulder = get_landmark(
                pose,
                RIGHT_SHOULDER
            )

            left_elbow = get_landmark(
                pose,
                LEFT_ELBOW
            )

            right_elbow = get_landmark(
                pose,
                RIGHT_ELBOW
            )

            left_wrist = get_landmark(
                pose,
                LEFT_WRIST
            )

            right_wrist = get_landmark(
                pose,
                RIGHT_WRIST
            )

            left_hip = get_landmark(
                pose,
                LEFT_HIP
            )

            right_hip = get_landmark(
                pose,
                RIGHT_HIP
            )

            left_knee = get_landmark(
                pose,
                LEFT_KNEE
            )

            right_knee = get_landmark(
                pose,
                RIGHT_KNEE
            )

            left_ankle = get_landmark(
                pose,
                LEFT_ANKLE
            )

            right_ankle = get_landmark(
                pose,
                RIGHT_ANKLE

            )

            left_heel = get_landmark(
                pose,
                LEFT_HEEL
            )

            right_heel = get_landmark(
                pose,
                RIGHT_HEEL
            )

            left_foot = get_landmark(
                pose,
                LEFT_FOOT_INDEX
            )

            right_foot = get_landmark(
                pose,
                RIGHT_FOOT_INDEX
            )


            # =================================================
            # SHOULDER
            # =================================================

            shoulders_visible = (

                landmark_visible(left_shoulder)

                and

                landmark_visible(right_shoulder)
            )


            if shoulders_visible:

                shoulder_status = "SHOULDER: SHOWN"


            # =================================================
            # HIP
            # =================================================

            hips_visible = (

                landmark_visible(left_hip)

                and

                landmark_visible(right_hip)
            )


            if hips_visible:

                hip_status = "HIP: SHOWN"


            # =================================================
            # LEFT KNEE
            # =================================================

            left_knee_reliable = (

                landmark_visible(left_hip)

                and

                landmark_visible(left_knee)

                and

                landmark_visible(left_ankle)
            )


            if left_knee_reliable:

                angle = calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )


                if angle is not None:

                    left_knee_status = (
                        f"LEFT KNEE: {int(angle)} deg"
                    )


                    if angle < KNEE_CAUTION_ANGLE:

                        warnings["left_knee"] = True


            # =================================================
            # RIGHT KNEE
            # =================================================

            right_knee_reliable = (

                landmark_visible(right_hip)

                and

                landmark_visible(right_knee)

                and

                landmark_visible(right_ankle)
            )


            if right_knee_reliable:

                angle = calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )


                if angle is not None:

                    right_knee_status = (
                        f"RIGHT KNEE: {int(angle)} deg"
                    )


                    if angle < KNEE_CAUTION_ANGLE:

                        warnings["right_knee"] = True


            # =================================================
            # LEFT ANKLE
            # =================================================

            left_ankle_reliable = (

                landmark_visible(left_knee)

                and

                landmark_visible(left_ankle)

                and

                (
                    landmark_visible(left_foot)
                    or
                    landmark_visible(left_heel)
                )
            )


            if left_ankle_reliable:

                left_ankle_status = (
                    "LEFT ANKLE: SHOWN"
                )


                if (
                    landmark_visible(left_knee)
                    and
                    landmark_visible(left_ankle)
                    and
                    landmark_visible(left_foot)
                ):

                    ankle_angle = calculate_angle(
                        left_knee,
                        left_ankle,
                        left_foot
                    )


                    if ankle_angle is not None:

                        if (
                            ankle_angle < ANKLE_LOW_ANGLE
                            or
                            ankle_angle > ANKLE_HIGH_ANGLE
                        ):

                            warnings["left_ankle"] = True


            # =================================================
            # RIGHT ANKLE
            # =================================================

            right_ankle_reliable = (

                landmark_visible(right_knee)

                and

                landmark_visible(right_ankle)

                and

                (
                    landmark_visible(right_foot)
                    or
                    landmark_visible(right_heel)
                )
            )


            if right_ankle_reliable:

                right_ankle_status = (
                    "RIGHT ANKLE: SHOWN"
                )


                if (
                    landmark_visible(right_knee)
                    and
                    landmark_visible(right_ankle)
                    and
                    landmark_visible(right_foot)
                ):

                    ankle_angle = calculate_angle(
                        right_knee,
                        right_ankle,
                        right_foot
                    )


                    if ankle_angle is not None:

                        if (
                            ankle_angle < ANKLE_LOW_ANGLE
                            or
                            ankle_angle > ANKLE_HIGH_ANGLE
                        ):

                            warnings["right_ankle"] = True


            # =================================================
            # LEFT ELBOW
            # =================================================

            left_elbow_reliable = (

                landmark_visible(left_shoulder)

                and

                landmark_visible(left_elbow)

                and

                landmark_visible(left_wrist)
            )


            if left_elbow_reliable:

                elbow_angle = calculate_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )


                if elbow_angle is not None:

                    left_elbow_status = (
                        f"LEFT ELBOW: "
                        f"{int(elbow_angle)} deg"
                    )


                    if elbow_angle < ELBOW_CAUTION_ANGLE:

                        warnings["left_elbow"] = True


            # =================================================
            # RIGHT ELBOW
            # =================================================

            right_elbow_reliable = (

                landmark_visible(right_shoulder)

                and

                landmark_visible(right_elbow)

                and

                landmark_visible(right_wrist)
            )


            if right_elbow_reliable:

                elbow_angle = calculate_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                )


                if elbow_angle is not None:

                    right_elbow_status = (
                        f"RIGHT ELBOW: "
                        f"{int(elbow_angle)} deg"
                    )


                    if elbow_angle < ELBOW_CAUTION_ANGLE:

                        warnings["right_elbow"] = True


            # =================================================
            # LEFT WRIST
            # =================================================

            left_wrist_reliable = (

                landmark_visible(left_elbow)

                and

                landmark_visible(left_wrist)
            )


            if left_wrist_reliable:

                left_wrist_status = (
                    "LEFT WRIST: SHOWN"
                )


            # =================================================
            # RIGHT WRIST
            # =================================================

            right_wrist_reliable = (

                landmark_visible(right_elbow)

                and

                landmark_visible(right_wrist)
            )


            if right_wrist_reliable:

                right_wrist_status = (
                    "RIGHT WRIST: SHOWN"
                )


            # =================================================
            # SHOULDER SCREENING
            # =================================================

            if shoulders_visible:

                shoulder_angle = calculate_angle(
                    left_hip,
                    left_shoulder,
                    right_shoulder
                ) if (
                    landmark_visible(left_hip)
                ) else None


                # Shoulder is primarily reported as shown
                # unless the geometry is clearly abnormal.

                if shoulder_angle is not None:

                    if shoulder_angle < SHOULDER_LOW_ANGLE:

                        warnings["shoulder"] = True


            # =================================================
            # TORSO SCREENING
            # =================================================

            torso_reliable = (

                landmark_visible(left_shoulder)

                and

                landmark_visible(right_shoulder)

                and

                landmark_visible(left_hip)

                and

                landmark_visible(right_hip)
            )


            if torso_reliable:

                shoulder_mid = midpoint(
                    left_shoulder,
                    right_shoulder
                )

                hip_mid = midpoint(
                    left_hip,
                    right_hip
                )


                torso_status = "TORSO: SHOWN"


                if shoulder_mid is not None and hip_mid is not None:

                    dx = (
                        shoulder_mid.x
                        -
                        hip_mid.x
                    )

                    dy = (
                        shoulder_mid.y
                        -
                        hip_mid.y
                    )


                    torso_angle = abs(
                        math.degrees(
                            math.atan2(
                                dx,
                                dy
                            )
                        )
                    )


                    if (
                        torso_angle
                        >
                        TORSO_LEAN_THRESHOLD
                    ):

                        warnings["torso"] = True


            # =================================================
            # FALL / SUDDEN BODY MOVEMENT SCREENING
            # =================================================

            if (
                landmark_visible(nose)
                and
                landmark_visible(left_hip)
                and
                landmark_visible(right_hip)
            ):

                hip_mid = midpoint(
                    left_hip,
                    right_hip
                )


                if hip_mid is not None:

                    current_body_y = (
                        nose.y * 0.4
                        +
                        hip_mid.y * 0.6
                    )


                    current_time = time.time()


                    if previous_body_y is not None:

                        vertical_change = (
                            current_body_y
                            -
                            previous_body_y
                        )


                        # Detect sudden downward movement.

                        if (
                            vertical_change
                            >
                            FALL_VERTICAL_CHANGE
                            and
                            fall_cooldown == 0
                        ):

                            warnings["fall"] = True

                            fall_status = (
                                "CAUTION: "
                                "SUDDEN DOWNWARD MOVEMENT"
                            )

                            fall_cooldown = (
                                FALL_COOLDOWN_FRAMES
                            )


                    previous_body_y = current_body_y


            if fall_cooldown > 0:

                fall_cooldown -= 1


            # =================================================
            # DRAW BODY
            # =================================================

            draw_body(
                frame,
                pose,
                width,
                height,
                warnings
            )


        # ====================================================
        # CREATE INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (620, 430),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # BODY
        # ====================================================

        if body_status == "BODY: DETECTED":

            body_color = (0, 255, 0)

        else:

            body_color = (255, 255, 255)


        put_text(
            frame,
            body_status,
            25,
            40,
            body_color
        )


        # ====================================================
        # KNEES
        # ====================================================

        left_color = (
            (0, 0, 255)
            if warnings["left_knee"]
            else (255, 255, 255)
        )


        right_color = (
            (0, 0, 255)
            if warnings["right_knee"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            left_knee_status,
            25,
            75,
            left_color
        )


        put_text(
            frame,
            right_knee_status,
            25,
            105,
            right_color
        )


        # ====================================================
        # ANKLES
        # ====================================================

        left_ankle_color = (
            (0, 0, 255)
            if warnings["left_ankle"]
            else (255, 255, 255)
        )


        right_ankle_color = (
            (0, 0, 255)
            if warnings["right_ankle"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            left_ankle_status,
            25,
            140,
            left_ankle_color
        )


        put_text(
            frame,
            right_ankle_status,
            25,
            170,
            right_ankle_color
        )


        # ====================================================
        # ELBOWS
        # ====================================================

        left_elbow_color = (
            (0, 0, 255)
            if warnings["left_elbow"]
            else (255, 255, 255)
        )


        right_elbow_color = (
            (0, 0, 255)
            if warnings["right_elbow"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            left_elbow_status,
            25,
            205,
            left_elbow_color
        )


        put_text(
            frame,
            right_elbow_status,
            25,
            235,
            right_elbow_color
        )


        # ====================================================
        # WRISTS
        # ====================================================

        left_wrist_color = (
            (0, 0, 255)
            if warnings["left_wrist"]
            else (255, 255, 255)
        )


        right_wrist_color = (
            (0, 0, 255)
            if warnings["right_wrist"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            left_wrist_status,
            25,
            270,
            left_wrist_color
        )


        put_text(
            frame,
            right_wrist_status,
            25,
            300,
            right_wrist_color
        )


        # ====================================================
        # SHOULDER
        # ====================================================

        shoulder_color = (
            (0, 0, 255)
            if warnings["shoulder"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            shoulder_status,
            25,
            335,
            shoulder_color
        )


        # ====================================================
        # HIP
        # ====================================================

        hip_color = (
            (0, 0, 255)
            if warnings["hip"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            hip_status,
            25,
            365,
            hip_color
        )


        # ====================================================
        # TORSO
        # ====================================================

        torso_color = (
            (0, 0, 255)
            if warnings["torso"]
            else (255, 255, 255)
        )


        put_text(
            frame,
            torso_status,
            25,
            395,
            torso_color
        )


        # ====================================================
        # CAUTION MESSAGES
        # ====================================================

        active_warnings = []


        if warnings["left_knee"]:

            active_warnings.append(
                "LEFT KNEE"
            )


        if warnings["right_knee"]:

            active_warnings.append(
                "RIGHT KNEE"
            )


        if warnings["left_ankle"]:

            active_warnings.append(
                "LEFT ANKLE"
            )


        if warnings["right_ankle"]:

            active_warnings.append(
                "RIGHT ANKLE"
            )


        if warnings["left_elbow"]:

            active_warnings.append(
                "LEFT ELBOW"
            )


        if warnings["right_elbow"]:

            active_warnings.append(
                "RIGHT ELBOW"
            )


        if warnings["shoulder"]:

            active_warnings.append(
                "SHOULDER"
            )


        if warnings["hip"]:

            active_warnings.append(
                "HIP"
            )


        if warnings["torso"]:

            active_warnings.append(
                "TORSO"
            )


        if warnings["fall"]:

            active_warnings.append(
                "SUDDEN MOVEMENT"
            )


        # ====================================================
        # DISPLAY WARNING
        # ====================================================

        if active_warnings:

            cv2.rectangle(
                frame,
                (10, 450),
                (900, 515),
                (0, 0, 0),
                -1
            )


            warning_text = (
                "CAUTION: "
                +
                ", ".join(active_warnings)
            )


            put_text(
                frame,
                warning_text,
                25,
                490,
                (0, 0, 255),
                0.70
            )


            put_text(
                frame,
                "Movement screening only - not an injury diagnosis",
                25,
                540,
                (255, 255, 255),
                0.48
            )


        elif body_status == "BODY: DETECTED":

            put_text(
                frame,
                "MOVEMENT SCREEN: ACTIVE",
                25,
                470,
                (0, 255, 0),
                0.65
            )


        # ====================================================
        # FALL STATUS
        # ====================================================

        if fall_status:

            put_text(
                frame,
                fall_status,
                25,
                575,
                (0, 0, 255),
                0.65
            )


        # ====================================================
        # LEGEND
        # ====================================================

        put_text(
            frame,
            "GREEN = tracked",
            width - 280,
            35,
            (0, 255, 0),
            0.50
        )


        put_text(
            frame,
            "RED = caution",
            width - 280,
            65,
            (0, 0, 255),
            0.50
        )


        # ====================================================
        # QUIT
        # ====================================================

        put_text(
            frame,
            "Press Q to quit",
            25,
            height - 25,
            (255, 255, 255),
            0.55
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
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

print("\nPlaySafe AI camera stopped.")