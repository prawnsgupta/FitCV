# FitCV — AI Fitness Tracker

[![CI](https://github.com/prawnsgupta/FitCV/actions/workflows/ci.yml/badge.svg)](https://github.com/prawnsgupta/FitCV/actions/workflows/ci.yml)

An AI-powered fitness tracking application that uses computer vision and deep learning to recognize exercises, count repetitions, and provide real-time posture correction using a webcam.

The project combines **MediaPipe Pose** for human pose estimation with a custom **PyTorch** model that integrates **Physics-Informed Neural Networks (PINN)**, **Spatio-Temporal Graph Convolutional Networks (ST-GCN)**, **Transformers**, and **LSTMs** to analyze both spatial body posture and temporal movement patterns.

---

## Features

- Real-time exercise recognition
- Automatic repetition counting
- Live posture correction and feedback
- Supports multiple exercises
- Web interface using FastAPI
- Standalone desktop application using OpenCV

### Supported Exercises

- Push-ups
- Squats
- Bicep Curls
- Lateral Raises
- Bent-Over Rows

---

## Model Architecture

The prediction pipeline consists of the following stages:

1. **Pose Estimation**
   - MediaPipe extracts 33 body landmarks from each video frame.

2. **Feature Extraction**
   - Joint angles and normalized skeleton coordinates are computed.

3. **PINN Layer**
   - Estimates velocity and acceleration while embedding biomechanical constraints into the model.

4. **ST-GCN**
   - Learns spatial relationships between connected body joints.

5. **Transformer + LSTM**
   - Captures temporal dependencies across sequences of movement.

6. **Prediction**
   - Classifies the exercise, counts repetitions, and generates posture correction feedback.

---

## Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- MediaPipe
- OpenCV
- FastAPI
- Pandas
- NumPy

---

## Installation

Clone the repository:

```
git clone https://github.com/prawnsgupta/FitCV.git
cd FitCV
```

Install the required packages:

```
pip install -r requirements.txt
```

---

## Running the Project

### Web Application

Start the FastAPI server:

```
uvicorn app:app --reload
```

Open your browser and visit:

```
http://localhost:8000
```

---

### Desktop Version

Run:

```
python main.py
```

Keyboard shortcuts:

| Key | Exercise |
|------|----------|
| P | Push-up |
| S | Squat |
| B | Bicep Curl |
| L | Lateral Raise |
| R | Bent-Over Row |
| Q | Quit |

---

## Collecting Training Data

To record your own exercise dataset:

```
python data_collection.py
```

Select an exercise mode, perform repetitions, and label the collected frames using the provided keyboard controls. The recorded samples are automatically saved for training.

---

## Training

Train a model for a specific exercise:

```
python train.py --mode Squat
```

The trained model weights will be saved in the `weights/` directory.

---

## Project Structure

```
AI-FITNESS-TRACKER/
│
├── app.py
├── main.py
├── model.py
├── train.py
├── dataset.py
├── data_collection.py
├── rep_counter.py
├── config.py
├── requirements.txt
├── dataset/
├── weights/
└── README.md
```

---

## Future Improvements

- Held-out per-clip validation split + reported test accuracy
- Additional exercise support
- Personalized workout analytics
- Mobile application
- Workout history and progress tracking
- Improved pose estimation and model accuracy

---

## Model card — shipped checkpoints

Five per-exercise checkpoints ship in `weights/`, trained on the recorded landmark
dataset in `dataset/` (~950 labelled clips). Verified with `offline_eval.py`
(headless, no webcam needed):

| Exercise | Classes | Windows | In-sample accuracy |
|---|---|---|---|
| Push-up | 5 | 856 | 100.0% |
| Squat | 3 | 372 | 99.7% |
| Bicep Curl | 4 | 936 | 100.0% |
| Lateral Raise | 3 | 882 | 100.0% |
| Bent-Over Row | 4 | 1028 | 100.0% |

> **Caveat:** `train.py` fits on all windows (no held-out split), so these are
> **in-sample** numbers — they verify that the shipped weights, feature pipeline
> (99 normalized coordinates + 12 biomechanical angles), and label maps are
> consistent end-to-end, not generalisation to unseen recordings. A held-out,
> per-clip split is the top item under Future Improvements.

Reproduce:

```
python offline_eval.py                  # all exercises
python offline_eval.py --mode Squat     # one exercise
```

---

## Attribution

The model architecture, dataset, and training pipeline were built by
[Soorya Natarajan](https://github.com/sooryanatarajan) (original repo:
AI-FITNESS-TRACKER). This fork adds productionization by
[Priyansh Gupta](https://github.com/prawnsgupta): the pinned dependency set
(see `requirements.txt` — newer MediaPipe drops the legacy `mp.solutions` Pose
API), FastAPI fixes, headless offline evaluation (`offline_eval.py`), shared
`features.py` (training/eval no longer initialise the MediaPipe camera stack),
CI, and documentation.

---

## Acknowledgements

This project is built using:

- MediaPipe Pose
- PyTorch
- PyTorch Geometric
- OpenCV
- FastAPI
