import cv2
import mediapipe as mp
import numpy as np
import os

# =============================
# Updated Absolute Paths
# =============================

# Base directory of your project
BASE_DIR = r"C:\Users\Admin\Downloads\Indian-Sign-Language-Recognition-pycode"

# Path to the dataset videos/frames
VIDEO_FOLDER = os.path.join(BASE_DIR, "test", "ISL_CSLRT_Corpus", "Frames_Word_Level")

# Output paths
OUTPUT_X = os.path.join(BASE_DIR, "test", "labels", "train_word_input_hands.npy")
OUTPUT_Y = os.path.join(BASE_DIR, "test", "labels", "train_word_labels.txt")

# =============================
# Mediapipe Initialization
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

X = []
y = []

# =============================
# Function to Extract Hand Keypoints
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
# Loop Through Each Class Folder
# =============================
for label in os.listdir(VIDEO_FOLDER):
    class_path = os.path.join(VIDEO_FOLDER, label)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing label: {label}")

    for video_file in os.listdir(class_path):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            kp = extract_hand_keypoints(results)

            X.append(kp)
            y.append(label)

        cap.release()

# =============================
# Save Outputs
# =============================
X = np.array(X)
y = np.array(y)

np.save(OUTPUT_X, X)
with open(OUTPUT_Y, "w") as f:
    for label in y:
        f.write(label + "\n")

print(f"Hand keypoints extracted and saved to:\n{OUTPUT_X}\n{OUTPUT_Y}")
