import cv2
import mediapipe as mp
import math

# ============================================================
# PLAYSAFE AI - POSE + JOINT ANGLE + MOVEMENT DETECTION
# ============================================================

MODEL_PATH = "ai/pose_landmarker_full.task"


# ------------------------------------------------------------
# Calculate angle between 3 landmarks
# ------------------------------------------------------------
def calculate_angle(a, b, c):
    """
    Calculates the angle at point b.

    a = first point
    b = middle/joint point
    c = third point
    """

    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x)
        - math.atan2(a.y - b.y, a.x - b.x)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ------------------------------------------------------------
# MediaPipe setup
# ------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE
)


# ------------------------------------------------------------
# Skeleton connections
# ------------------------------------------------------------

POSE_CONNECTIONS = [

    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Mouth
    (9, 10),

    # Shoulders
    (11, 12),

    # Right arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),

    # Left arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Right leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),

    # Left leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32)
]


# ------------------------------------------------------------
# Start Pose Landmarker
# ------------------------------------------------------------

with PoseLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        exit()

    print("--------------------------------")
    print("       PLAYSAFE AI STARTED")
    print("--------------------------------")
    print("Camera: OK")
    print("Pose Detection: OK")
    print("Press Q to quit.")
    print("--------------------------------")


    # Previous elbow angle
    previous_elbow_angle = None


    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------

    while True:

        ret, frame = camera.read()


        if not ret:

            print("ERROR: Could not read camera frame.")

            break


        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Create MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # Detect pose
        # ----------------------------------------------------

        result = landmarker.detect(mp_image)


        # ----------------------------------------------------
        # If a person is detected
        # ----------------------------------------------------

        if result.pose_landmarks:

            for pose_landmarks in result.pose_landmarks:

                height, width, _ = frame.shape


                # =================================================
                # RIGHT ARM
                # =================================================

                right_shoulder = pose_landmarks[12]

                right_elbow = pose_landmarks[14]

                right_wrist = pose_landmarks[16]


                # -------------------------------------------------
                # Calculate right elbow angle
                # -------------------------------------------------

                elbow_angle = calculate_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                )


                # -------------------------------------------------
                # Calculate angle change
                # -------------------------------------------------

                if previous_elbow_angle is not None:

                    angle_change = abs(
                        elbow_angle
                        - previous_elbow_angle
                    )

                else:

                    angle_change = 0


                # Save current angle
                previous_elbow_angle = elbow_angle


                # =================================================
                # DRAW LANDMARKS
                # =================================================

                for landmark in pose_landmarks:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    # Only draw landmarks inside screen
                    if (
                        0 <= x < width
                        and
                        0 <= y < height
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )


                # =================================================
                # DRAW SKELETON
                # =================================================

                for start, end in POSE_CONNECTIONS:

                    start_landmark = pose_landmarks[start]

                    end_landmark = pose_landmarks[end]


                    x1 = int(
                        start_landmark.x * width
                    )

                    y1 = int(
                        start_landmark.y * height
                    )


                    x2 = int(
                        end_landmark.x * width
                    )

                    y2 = int(
                        end_landmark.y * height
                    )


                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


                # =================================================
                # DISPLAY ELBOW ANGLE
                # =================================================

                cv2.putText(
                    frame,
                    f"Right Elbow: {int(elbow_angle)} deg",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                # =================================================
                # DISPLAY ANGLE CHANGE
                # =================================================

                cv2.putText(
                    frame,
                    f"Movement: {int(angle_change)} deg/frame",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


                # =================================================
                # MOVEMENT DETECTION
                # =================================================

                if angle_change > 15:

                    cv2.putText(
                        frame,
                        "FAST MOVEMENT DETECTED",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "NORMAL MOVEMENT",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


        else:

            # =================================================
            # NO PERSON DETECTED
            # =================================================

            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # =====================================================
        # DISPLAY CAMERA
        # =====================================================

        cv2.imshow(
            "PlaySafe AI - Movement Detection",
            frame
        )


        # =====================================================
        # PRESS Q TO QUIT
        # =====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()


print("PlaySafe AI stopped.")