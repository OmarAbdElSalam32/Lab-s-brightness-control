import cv2
import numpy as np
import screen_brightness_control as sbc
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get coordinates of the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert coordinates to pixel values
                h, w, c = frame.shape
                thumb_tip = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_tip = (int(index_tip.x * w), int(index_tip.y * h))

                # Draw circles on the thumb tip and index finger tip
                cv2.circle(frame, thumb_tip, 10, (0, 255, 0), -1)
                cv2.circle(frame, index_tip, 10, (0, 255, 0), -1)

                # Calculate the distance between thumb tip and index finger tip
                distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                # Map the distance to a brightness value (0 to 100)
                max_distance = 200  # Adjust this value as needed
                brightness = np.clip((distance / max_distance) * 100, 0, 100)

                # Set the screen brightness
                sbc.set_brightness(int(brightness))

                # Display the distance and brightness
                cv2.putText(frame, f'Distance: {int(distance)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Brightness: {int(brightness)}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()