import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# =============================
# Base Project Directory
# =============================
BASE_DIR = r"C:\Users\Admin\Downloads\Indian-Sign-Language-Recognition-pycode"

# =============================
# Model and Label Paths
# =============================
MODEL_PATH = os.path.join(BASE_DIR, "test", "model", "word_model_hands.keras")
LABELS_PATH = os.path.join(BASE_DIR, "test", "labels", "word_labels_nodes.npy")

# Load model and labels
print("Loading model and labels...")
word_model = load_model(MODEL_PATH)
word_labels = np.load(LABELS_PATH)
print("Model and labels loaded successfully!")

# =============================
# Initialize Mediapipe Hands
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# =============================
# Extract Hand Keypoints
# =============================
def extract_hand_keypoints(results):
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if i == 0:
                left_hand = hand
            elif i == 1:
                right_hand = hand
    return np.concatenate([left_hand, right_hand])

# =============================
# Start Webcam for Real-time Prediction
# =============================
cap = cv2.VideoCapture(0)  # Use webcam
prev_time = 0
interval = 1  # Predict every 1 second
pred_word = "..."

print("Starting real-time sign language recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Predict every 1 second
    if time.time() - prev_time > interval:
        kp_input = extract_hand_keypoints(results).reshape(1, -1)
        if kp_input.sum() != 0:  # Ensure hand keypoints detected
            prediction = word_model.predict(kp_input, verbose=0)
            pred_idx = np.argmax(prediction)
            pred_word = str(word_labels[pred_idx])
        else:
            pred_word = "No Hands Detected"
        prev_time = time.time()

    # Display predicted word
    cv2.putText(frame,
                f'Word: {pred_word}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Show the frame
    cv2.imshow("Hand Word Prediction", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# Cleanup
# =============================
cap.release()
cv2.destroyAllWindows()
print("Real-time recognition stopped.")
