import cv2
import mediapipe as mp
import numpy as np
import time


# -----------------------------
# EAR CALCULATION FUNCTION
# -----------------------------
def calculate_EAR(eye_points):

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)

    horizontal = np.linalg.norm(p1 - p4)

    EAR = (vertical1 + vertical2) / (2.0 * horizontal)

    return EAR


# -----------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

mp_draw = mp.solutions.drawing_utils


# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------------
# START WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.20
blink_start_time = 0
blink_counter = 0
morse_sequence = ""


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)


    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye_points = []
            right_eye_points = []


            # LEFT EYE
            for id in LEFT_EYE:

                x = int(face_landmarks.landmark[id].x * w)
                y = int(face_landmarks.landmark[id].y * h)

                left_eye_points.append(np.array([x, y]))

                cv2.circle(frame, (x, y), 3, (0,255,0), -1)


            # RIGHT EYE
            for id in RIGHT_EYE:

                x = int(face_landmarks.landmark[id].x * w)
                y = int(face_landmarks.landmark[id].y * h)

                right_eye_points.append(np.array([x, y]))

                cv2.circle(frame, (x, y), 3, (0,255,0), -1)


            # CALCULATE EAR
            left_EAR = calculate_EAR(left_eye_points)
            right_EAR = calculate_EAR(right_eye_points)

            EAR = (left_EAR + right_EAR) / 2.0


            # -----------------------------
            # BLINK DETECTION
            # -----------------------------
            if EAR < EAR_THRESHOLD:

                if blink_counter == 0:
                    blink_start_time = time.time()

                blink_counter += 1

            else:

                if blink_counter > 2:

                    blink_duration = time.time() - blink_start_time

                    if blink_duration < 0.5:
                        morse_sequence += "."
                        print("DOT")

                    else:
                        morse_sequence += "-"
                        print("DASH")

                    print("Current Morse:", morse_sequence)

                blink_counter = 0


            # DISPLAY EAR
            cv2.putText(
                frame,
                f"EAR: {EAR:.2f}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )


    cv2.imshow("Blink Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()