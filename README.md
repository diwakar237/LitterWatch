# LitterWatch
An AI-powered system that analyses video footage to detect littering behaviour using **YOLOv8**, **MediaPipe Pose Estimation**, and **FastAPI**.

## 🔍 Features
- Video upload via web interface
- Person & object detection (YOLOv8)
- Throw-gesture recognition (MediaPipe)
- Timestamped incident log
- Activity heatmap generation
- PDF incident report export

## 🗂️ Project Structure
\```
smart-littering-detection/
├── backend/        # FastAPI server + detection pipeline
├── frontend/       # HTML/CSS/JS web interface
├── colab/          # Google Colab setup notebook
\```

## 🚀 Quick Start (Local)
\```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
\```
Then open `frontend/littering_detection.html` in your browser.

## ☁️ Run on Google Colab
Open `colab/colab_setup.py` and follow the cells step by step.

## 🛠️ Tech Stack
- Python, FastAPI, OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- ReportLab (PDF)
- HTML / CSS / JavaScript (Frontend)
