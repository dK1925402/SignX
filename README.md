# Indian Sign Language Recognition 🤟

````markdown


Real-Time AI System for Indian Sign Language Detection using Python, MediaPipe, and TensorFlow/Keras.

---

## Features

- Detects left and right hand landmarks in real-time using **MediaPipe**.
- Trains a deep learning model on extracted hand keypoints.
- Provides real-time predictions using your webcam.
- Modular workflow: Preprocessing → Training → Real-Time Prediction.

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repository_url>
cd Indian-Sign-Language-Recognition-pycode
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### requirements.txt

```text
numpy>=1.24.0
opencv-python>=4.9.0.80
mediapipe>=0.10.15
tensorflow>=2.15.0
scikit-learn>=1.4.0
```

---

## Prepare Dataset

If you have the ISL dataset, place videos inside:

```
test/ISL_CSLRT_Corpus/Frames_Word_Level/
```

**Example folder structure:**

```
Frames_Word_Level/
├── Hello/
├── ThankYou/
├── ...
```

### Running Without Dataset

If you **don’t have the dataset**, you can create dummy files:

```python
import numpy as np

# Dummy keypoints for 10 samples (42*3*2 = 252 features)
np.save("test/labels/train_word_input_hands.npy", np.zeros((10, 126)))

# Dummy labels
with open("test/labels/train_word_labels.txt", "w") as f:
    f.write("\n".join(["dummy"]*10))

# Dummy label encoder
np.save("test/labels/word_labels_nodes.npy", np.array(["dummy"]))
```

This will allow the scripts to **run without errors**, even without the original dataset.

---

## Running the Scripts

1. **Extract Keypoints**

```bash
python test/utils/extract_keypoints.py
```

2. **Train Model**

```bash
python test/train_word_model_from_scratch.py
```

3. **Real-Time Prediction**

```bash
python test/realtime_hand_predict.py
```

Press `q` to quit the window.

---

## Notes

* Make sure Python 3.10+ is installed.
* If the webcam does not open, try changing the device index:

```python
cap = cv2.VideoCapture(1)
```

* If predictions are random, train the model with a proper dataset.

---

## Tech Stack

* Python 3.10+
* TensorFlow / Keras
* MediaPipe
* OpenCV
* Scikit-learn

---

## Future Improvements

* Support sentence-level ISL recognition.
* Deploy as web or mobile application.
* Integrate text-to-speech for accessibility.

````

---

✅ **Explanation:**  
- The entire README is written in Markdown syntax (`.md`).  
- **Code blocks** are wrapped with triple backticks ``` for shell commands, Python snippets, and text blocks.  
- It includes **requirements.txt**, dataset instructions, dummy dataset example, and running instructions.  

---

If you want, I can make a **super compact version** that only **shows requirements + dummy dataset code**, so it’s extremely minimal for quick usage.  

Do you want me to do that?
````
