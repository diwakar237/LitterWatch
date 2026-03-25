# ============================================================
# Smart Littering Detection System — Google Colab Setup
# Run each cell in order to set up and test the full pipeline
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install fastapi uvicorn python-multipart ultralytics mediapipe reportlab --quiet
# !pip install nest-asyncio --quiet   # needed to run uvicorn inside Colab

# ── CELL 2: Upload project files ────────────────────────────
# Upload main.py, detector.py, schemas.py, report.py
# to /content/ in Colab, then run this cell.

import os, nest_asyncio, uvicorn, threading

nest_asyncio.apply()
os.chdir("/content")

def run_server():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

t = threading.Thread(target=run_server, daemon=True)
t.start()
print("Server running at http://localhost:8000")

# ── CELL 3: Quick smoke-test ─────────────────────────────────
import requests, json, time

# Upload a test video (replace with your own)
VIDEO_PATH = "/content/sample_video.mp4"   # <-- change this

with open(VIDEO_PATH, "rb") as f:
    resp = requests.post(
        "http://localhost:8000/analyse",
        files={"file": ("video.mp4", f, "video/mp4")},
        data={"confidence": 0.55, "track_persons": True, "generate_heatmap": True},
    )

job = resp.json()
print(f"Job queued: {job['job_id']}")

# ── CELL 4: Poll for result ──────────────────────────────────
JOB_ID = job["job_id"]

while True:
    status = requests.get(f"http://localhost:8000/status/{JOB_ID}").json()
    print(f"[{status['progress']:3d}%] {status['status']} — {status.get('stage','')}")
    if status["status"] in ("done", "error"):
        break
    time.sleep(1.5)

if status["status"] == "done":
    result = status["result"]
    print("\n── RESULT ──────────────────────────────────────")
    print(f"Verdict           : {result['verdict']}")
    print(f"Littering events  : {result['event_count']}")
    print(f"Persons detected  : {result['persons_detected']}")
    print(f"Frames analysed   : {result['frames_analysed']}")
    print(f"Avg. confidence   : {result['avg_confidence']*100:.1f}%")
    print("\nEvent log:")
    for ev in result["events"]:
        print(f"  [{ev['timestamp_str']}]  conf={ev['confidence']:.2f}  {ev['description']}")

# ── CELL 5: Download PDF report ──────────────────────────────
from google.colab import files as colab_files

r = requests.get(f"http://localhost:8000/report/{JOB_ID}")
with open("littering_report.pdf", "wb") as f:
    f.write(r.content)

colab_files.download("littering_report.pdf")
print("Report downloaded!")
