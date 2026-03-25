"""
Smart Littering Detection System — FastAPI Backend
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid
import os
import shutil
from pathlib import Path
from typing import Optional

from detector import LitteringDetector
from report import generate_pdf_report
from schemas import AnalysisResult, JobStatus

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Littering Detection API",
    description="Upload a video and detect littering behaviour using AI.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ────────────────────────────────────────────────────────────────
UPLOAD_DIR  = Path("uploads")
RESULT_DIR  = Path("results")
REPORT_DIR  = Path("reports")

for d in [UPLOAD_DIR, RESULT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory job store (use Redis/DB in production)
jobs: dict[str, dict] = {}

# ── Detector singleton ─────────────────────────────────────────────────────────
detector = LitteringDetector(
    confidence_threshold=0.50,
    yolo_model="yolov8n.pt",        # auto-downloaded on first run
    use_pose=True,
)

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "LitterWatch API is running. POST /analyse to begin."}


@app.post("/analyse", response_model=dict)
async def analyse_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = 0.50,
    track_persons: bool = True,
    generate_heatmap: bool = True,
):
    """
    Accept a video upload, queue analysis as a background task,
    and return a job_id to poll for results.
    """
    # Validate file type
    allowed = {"video/mp4", "video/quicktime", "video/x-msvideo",
               "video/x-matroska", "video/webm"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    job_id   = str(uuid.uuid4())
    ext      = Path(file.filename).suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    # Stream file to disk
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "progress": 0, "result": None}

    # Run detection in background so upload returns immediately
    background_tasks.add_task(
        run_analysis,
        job_id=job_id,
        video_path=save_path,
        confidence=confidence,
        track_persons=track_persons,
        generate_heatmap=generate_heatmap,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    """Poll analysis progress."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        result=j.get("result"),
    )


@app.get("/report/{job_id}")
def download_report(job_id: str):
    """Download the generated PDF report."""
    report_path = REPORT_DIR / f"{job_id}.pdf"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not ready yet")
    return FileResponse(report_path, media_type="application/pdf",
                        filename=f"littering_report_{job_id[:8]}.pdf")


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Clean up files for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    # Remove uploaded video
    for f in UPLOAD_DIR.glob(f"{job_id}*"):
        f.unlink(missing_ok=True)
    for f in RESULT_DIR.glob(f"{job_id}*"):
        f.unlink(missing_ok=True)
    for f in REPORT_DIR.glob(f"{job_id}*"):
        f.unlink(missing_ok=True)
    del jobs[job_id]
    return {"message": "Job deleted"}


# ── Background Task ────────────────────────────────────────────────────────────

def run_analysis(job_id, video_path, confidence, track_persons, generate_heatmap):
    """Run the full detection pipeline and update job store."""
    try:
        jobs[job_id]["status"] = "processing"
        detector.confidence_threshold = confidence

        def progress_cb(pct: int, stage: str):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["stage"] = stage

        result: AnalysisResult = detector.analyse(
            video_path=str(video_path),
            job_id=job_id,
            result_dir=str(RESULT_DIR),
            progress_callback=progress_cb,
            track_persons=track_persons,
            generate_heatmap=generate_heatmap,
        )

        # Generate PDF report
        generate_pdf_report(result, str(REPORT_DIR / f"{job_id}.pdf"))

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"]   = result.dict()

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        raise
