import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw face mesh
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

            h, w, _ = frame.shape

            # Draw left eye points
            for id in LEFT_EYE:
                x = int(face_landmarks.landmark[id].x * w)
                y = int(face_landmarks.landmark[id].y * h)
                cv2.circle(frame, (x, y), 3, (0,255,0), -1)

            # Draw right eye points
            for id in RIGHT_EYE:
                x = int(face_landmarks.landmark[id].x * w)
                y = int(face_landmarks.landmark[id].y * h)
                cv2.circle(frame, (x, y), 3, (0,255,0), -1)

    # Show frame
    cv2.imshow("Eye Landmark Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()