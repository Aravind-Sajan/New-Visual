import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load your trained ASL model
model = tf.keras.models.load_model('models/asl_gesture_model.h5')

# Define the class labels (ASL alphabet)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the image for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract the 21 landmarks (x, y) coordinates for each hand
            hand_points = []
            for landmark in landmarks.landmark:
                hand_points.append([landmark.x, landmark.y])

            # Convert to a numpy array and normalize the coordinates
            hand_points = np.array(hand_points).flatten()
            hand_points = hand_points / np.max(hand_points)  # Normalize to 0-1 range

            # Reshape for the model input
            hand_points = np.expand_dims(hand_points, axis=0)

            # Predict the gesture from the model
            prediction = model.predict(hand_points)
            predicted_class = np.argmax(prediction, axis=1)

            # Display the predicted gesture
            predicted_letter = class_labels[predicted_class[0]]

            # Draw landmarks and the predicted letter on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("ASL Gesture Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
