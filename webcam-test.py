import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw coordinates on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print coordinates of the hand
            for i, lm in enumerate(hand_landmarks.landmark):
                if i < 3:
                    print(f"Landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

    cv2.imshow("ASL Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
