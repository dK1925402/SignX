<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Indian Sign Language Recognition - README</title>
  <style>
    body {
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f4f4f4;
      margin: 0;
      padding: 20px;
      line-height: 1.6;
      color: #333;
    }
    h1, h2, h3 {
      color: #2c7a7b;
    }
    h1 {
      border-bottom: 3px solid #2c7a7b;
      padding-bottom: 5px;
    }
    code, pre {
      background-color: #eaeaea;
      padding: 5px 10px;
      border-radius: 5px;
      font-family: monospace;
    }
    pre {
      overflow-x: auto;
    }
    ul, ol {
      padding-left: 20px;
    }
    section {
      margin-bottom: 20px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 20px;
    }
    table, th, td {
      border: 1px solid #ccc;
    }
    th, td {
      padding: 10px;
      text-align: left;
    }
    .note {
      background-color: #fff3cd;
      border-left: 5px solid #ffc107;
      padding: 10px;
      margin-bottom: 10px;
      border-radius: 4px;
    }
    footer {
      text-align: center;
      margin-top: 40px;
      padding: 15px 0;
      background: #2c7a7b;
      color: white;
      border-radius: 5px;
    }
  </style>
</head>
<body>

  <h1>Indian Sign Language Recognition 🤟</h1>

  <section>
    <h2>Features</h2>
    <ul>
      <li>Detects left and right hand landmarks in real-time using <strong>MediaPipe</strong>.</li>
      <li>Trains a <strong>deep learning model</strong> on extracted keypoints.</li>
      <li>Provides real-time predictions via webcam.</li>
      <li>Modular workflow: Preprocessing → Training → Real-Time Prediction.</li>
    </ul>
  </section>

  <section>
    <h2>Folder Structure</h2>
    <pre>
Indian-Sign-Language-Recognition-pycode
│
├── requirements.txt
├── README.html
│
└── test
    ├── ISL_CSLRT_Corpus
    │   └── Frames_Word_Level
    │       ├── Hello
    │       └── ThankYou
    │
    ├── labels
    │   ├── train_word_input_hands.npy
    │   ├── train_word_labels.txt
    │   └── word_labels_nodes.npy
    │
    ├── model
    │   └── word_model_hands.keras
    │
    ├── utils
    │   └── extract_keypoints.py
    │
    ├── realtime_hand_predict.py
    └── train_word_model_from_scratch.py
    </pre>
  </section>

  <section>
    <h2>Setup Instructions</h2>

    <h3>1. Clone the repository</h3>
    <pre><code>git clone &lt;repository_url&gt;
cd Indian-Sign-Language-Recognition-pycode</code></pre>

    <h3>2. Install dependencies</h3>
    <pre><code>pip install -r requirements.txt</code></pre>

    <h4>requirements.txt</h4>
    <pre><code>
numpy>=1.24.0
opencv-python>=4.9.0.80
mediapipe>=0.10.15
tensorflow>=2.15.0
scikit-learn>=1.4.0
    </code></pre>

    <h3>3. Prepare Dataset</h3>
    <p>Place your ISL dataset videos inside:</p>
    <pre>test/ISL_CSLRT_Corpus/Frames_Word_Level/</pre>

    <h3>4. Extract Keypoints</h3>
    <pre><code>python test/utils/extract_keypoints.py</code></pre>

    <h3>5. Train Model</h3>
    <pre><code>python test/train_word_model_from_scratch.py</code></pre>

    <h3>6. Real-Time Prediction</h3>
    <pre><code>python test/realtime_hand_predict.py</code></pre>
    <p>Press <strong>q</strong> to quit the window.</p>
  </section>

  <section>
    <h2>Usage Flow</h2>
    <table>
      <tr>
        <th>Step</th>
        <th>Script</th>
        <th>Output</th>
      </tr>
      <tr>
        <td>1</td>
        <td>extract_keypoints.py</td>
        <td>Keypoints `.npy` + labels `.txt`</td>
      </tr>
      <tr>
        <td>2</td>
        <td>train_word_model_from_scratch.py</td>
        <td>Trained `.keras` model + label encoder `.npy`</td>
      </tr>
      <tr>
        <td>3</td>
        <td>realtime_hand_predict.py</td>
        <td>Live sign language predictions</td>
      </tr>
    </table>
  </section>

  <section>
    <h2>Run Without Dataset</h2>
    <p>If you don't have a dataset, create dummy files:</p>
    <pre><code>import numpy as np
np.save("test/labels/train_word_input_hands.npy", np.zeros((10, 126)))
with open("test/labels/train_word_labels.txt", "w") as f:
    f.write("\n".join(["dummy"]*10))
np.save("test/labels/word_labels_nodes.npy", np.array(["dummy"]))</code></pre>
  </section>

  <section>
    <h2>Troubleshooting</h2>
    <ul>
      <li><strong>FileNotFoundError:</strong> Run preprocessing and training scripts first.</li>
      <li><strong>Webcam not opening:</strong> Check permissions or use cv2.VideoCapture(1).</li>
      <li><strong>Random predictions:</strong> Model not trained or dataset too small.</li>
      <li><strong>Slow performance:</strong> Use a system with GPU support.</li>
    </ul>
  </section>

  <section>
    <h2>Tech Stack</h2>
    <ul>
      <li>Python 3.10+</li>
      <li>TensorFlow / Keras</li>
      <li>MediaPipe</li>
      <li>OpenCV</li>
      <li>Scikit-learn</li>
    </ul>
  </section>

  <section>
    <h2>Future Improvements</h2>
    <ul>
      <li>Support sentence-level ISL recognition.</li>
      <li>Deploy as web or mobile app.</li>
      <li>Integrate text-to-speech for accessibility.</li>
    </ul>
  </section>

  <footer>
    <p>Indian Sign Language Recognition Project &copy; 2025 | Open Source</p>
  </footer>

</body>
</html>
