
Lip Reading using Deep Learning

This project implements a Lip Reading System that recognizes spoken words directly from lip movements using Deep Learning techniques.

🚀 Project Overview

Extracts frames at 25 FPS from 3-second video clips using OpenCV.

Data Preprocessing:

Converted frames from RGB → Grayscale.

Cropped lip region from each frame to focus on the mouth area.

Processes video data with a 3D CNN + BiLSTM architecture.

Trains with CTC Loss to align predicted sequences with target labels.

Achieved:

99.51% accuracy at the character level

98.67% accuracy at the word level

Dataset: GRID Corpus

🧑‍💻 Tech Stack

Backend: FastAPI (fastapi/)

Frontend: React.js (deployed on Vercel)

Training & Testing:

train.py → model training

test.py → model evaluation

GPU Training: Fully trained on NVIDIA A100 (Google Colab)

Project Structure
├── Fastapi/          # Backend (FastAPI)
├── frontend/         # Frontend (React.js, deployed on Vercel)
├── train.py          # Training script
├── test.py           # Testing script
└── README.md         # Project documentation

Deployment Notes

The trained model file (.h5) is not included in this repository due to large file size limitations.

The system requires a GPU server for real-time inference and thus cannot be deployed fully online.

The frontend is live on Vercel, but backend deployment is pending due to GPU support.

Results

Character-level Accuracy: 99.51%

Word-level Accuracy: 98.67%

Future Work

Optimize for deployment on GPU-based servers.

Experiment with transformers for better sequence modeling.

Extend to larger datasets for real-world scenarios.

