"""
detector.py — Core littering detection pipeline

Pipeline stages:
  1. Frame extraction (OpenCV)
  2. Person & object detection (YOLOv8)
  3. Pose estimation (MediaPipe)
  4. Throw-gesture analysis (heuristic + angle scoring)
  5. Activity heatmap generation (Gaussian blobs)
  6. Result compilation
"""

import cv2
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict

from schemas import AnalysisResult, LitteringEvent, PersonTrack, HeatmapData

# Optional imports — graceful fallback if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Using mock detections.")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARN] mediapipe not installed. Pose estimation disabled.")


# ── Constants ──────────────────────────────────────────────────────────────────

# COCO class IDs that count as "litter" if detected on the ground
LITTER_CLASSES = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    67: "cell phone",
    73: "book",
    74: "clock",
    76: "scissors",
    79: "toothbrush",
}

# Frame-skip interval (analyse every Nth frame for speed)
FRAME_SKIP = 3

# Minimum wrist drop (px, normalised) to count as a throw gesture
WRIST_DROP_THRESHOLD = 0.08

# Gaussian spread for heatmap blobs
BLOB_SIGMA = 40


# ── Main Detector Class ────────────────────────────────────────────────────────

class LitteringDetector:
    def __init__(
        self,
        confidence_threshold: float = 0.50,
        yolo_model: str = "yolov8n.pt",
        use_pose: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.use_pose = use_pose and MP_AVAILABLE

        # Load YOLO
        if YOLO_AVAILABLE:
            print(f"[INFO] Loading YOLO model: {yolo_model}")
            self.yolo = YOLO(yolo_model)
        else:
            self.yolo = None

        # Load MediaPipe Pose
        if self.use_pose:
            self.mp_pose    = mp.solutions.pose
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.pose_model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyse(
        self,
        video_path: str,
        job_id: str,
        result_dir: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        track_persons: bool = True,
        generate_heatmap: bool = True,
    ) -> AnalysisResult:

        def cb(pct, stage):
            if progress_callback:
                progress_callback(pct, stage)

        cb(5, "frame_extraction")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps

        print(f"[INFO] Video: {width}x{height}, {fps:.1f} fps, {total_frames} frames, {duration_s:.1f}s")

        # ── Heatmap accumulator ────────────────────────────────────────────────
        heatmap_acc = np.zeros((height, width), dtype=np.float32)

        # ── State ──────────────────────────────────────────────────────────────
        events: list[LitteringEvent] = []
        person_tracks: dict[int, PersonTrack] = {}
        analysed_frames = 0
        frame_idx = 0

        # ── Frame loop ─────────────────────────────────────────────────────────
        cb(10, "object_detection")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames for speed
            if frame_idx % FRAME_SKIP != 0:
                continue

            analysed_frames += 1
            timestamp_s = frame_idx / fps
            pct = 10 + int(70 * frame_idx / max(total_frames, 1))
            cb(min(pct, 79), "behaviour_analysis")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── YOLO detection ─────────────────────────────────────────────────
            detections = self._run_yolo(frame_rgb)

            person_boxes   = [d for d in detections if d["class"] == 0]    # class 0 = person
            litter_objects = [d for d in detections if d["class"] in LITTER_CLASSES]

            # ── Update person tracks ───────────────────────────────────────────
            if track_persons:
                self._update_tracks(person_tracks, person_boxes, frame_idx, timestamp_s)

            # ── Pose estimation ────────────────────────────────────────────────
            throw_detected = False
            throw_confidence = 0.0
            throw_location  = None

            if self.use_pose and len(person_boxes) > 0:
                result = self.pose_model.process(frame_rgb)
                if result.pose_landmarks:
                    score, wrist_xy = self._score_throw_gesture(
                        result.pose_landmarks, width, height
                    )
                    if score > WRIST_DROP_THRESHOLD:
                        throw_detected   = True
                        throw_confidence = min(1.0, score * 4.0)
                        throw_location   = wrist_xy

            # ── Heuristic fallback (YOLO only) ─────────────────────────────────
            # If a litter object appears near a person's feet, flag it
            if not throw_detected and litter_objects and person_boxes:
                score = self._proximity_score(person_boxes, litter_objects, width, height)
                if score > self.confidence_threshold:
                    throw_detected   = True
                    throw_confidence = score
                    # Use centroid of nearest litter object
                    lo = litter_objects[0]["bbox"]
                    throw_location = (
                        int((lo[0] + lo[2]) / 2),
                        int((lo[1] + lo[3]) / 2),
                    )

            # ── Record event ───────────────────────────────────────────────────
            if throw_detected and throw_confidence >= self.confidence_threshold:
                events.append(LitteringEvent(
                    timestamp_s=round(timestamp_s, 2),
                    timestamp_str=self._fmt_time(timestamp_s),
                    confidence=round(throw_confidence, 3),
                    frame_number=frame_idx,
                    location_xy=list(throw_location) if throw_location else None,
                    description=self._describe_event(throw_confidence),
                ))
                # Accumulate heatmap
                if throw_location and generate_heatmap:
                    self._add_blob(heatmap_acc, throw_location[0], throw_location[1])

            # Ambient person heatmap (lighter)
            if generate_heatmap:
                for pb in person_boxes:
                    cx = int((pb["bbox"][0] + pb["bbox"][2]) / 2)
                    cy = int((pb["bbox"][1] + pb["bbox"][3]) / 2)
                    self._add_blob(heatmap_acc, cx, cy, strength=0.15)

        cap.release()
        cb(82, "heatmap_generation")

        # ── Post-process heatmap ───────────────────────────────────────────────
        heatmap_data: Optional[HeatmapData] = None
        if generate_heatmap:
            heatmap_data = self._encode_heatmap(
                heatmap_acc, width, height, job_id, result_dir
            )

        cb(92, "report_compile")

        # ── Build result ───────────────────────────────────────────────────────
        avg_conf = (
            sum(e.confidence for e in events) / len(events)
            if events else 0.0
        )

        result = AnalysisResult(
            job_id=job_id,
            verdict="LITTERING_DETECTED" if events else "CLEAN",
            littering_detected=bool(events),
            event_count=len(events),
            events=events,
            frames_analysed=analysed_frames,
            total_frames=total_frames,
            persons_detected=len(person_tracks),
            duration_s=round(duration_s, 2),
            avg_confidence=round(avg_conf, 3),
            fps=round(fps, 2),
            resolution=f"{width}x{height}",
            person_tracks=list(person_tracks.values()),
            heatmap=heatmap_data,
        )

        cb(100, "done")
        return result

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _run_yolo(self, frame_rgb: np.ndarray) -> list[dict]:
        """Run YOLOv8 and return normalised detections."""
        if not self.yolo:
            return self._mock_detections(frame_rgb.shape)
        results = self.yolo(frame_rgb, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append({
                "class": int(box.cls[0]),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
        return detections

    def _mock_detections(self, shape) -> list[dict]:
        """Return synthetic detections when YOLO is not available (for testing)."""
        h, w = shape[:2]
        if np.random.random() > 0.85:
            return [
                {"class": 0,  "confidence": 0.85, "bbox": [w//4, h//4, w//2, 3*h//4]},
                {"class": 39, "confidence": 0.72, "bbox": [w//3, 2*h//3, w//3+40, 2*h//3+60]},
            ]
        return [{"class": 0, "confidence": 0.80, "bbox": [w//4, h//4, w//2, 3*h//4]}]

    def _score_throw_gesture(self, landmarks, width: int, height: int):
        """
        Score the probability of a throw gesture from MediaPipe pose landmarks.
        Returns (score: float, wrist_xy: tuple).

        Key joints:
          RIGHT_WRIST=16, LEFT_WRIST=15
          RIGHT_HIP=24,   LEFT_HIP=23
          RIGHT_KNEE=26,  LEFT_KNEE=25
        """
        lm = landmarks.landmark
        mp_pose = self.mp_pose

        # Get key joint positions (normalised 0-1)
        try:
            r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            r_hip   = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            l_hip   = lm[mp_pose.PoseLandmark.LEFT_HIP]
            r_knee  = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            l_knee  = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        except (IndexError, AttributeError):
            return 0.0, None

        # Score 1: wrist is near or below the knee level (dropping motion)
        knee_y  = (r_knee.y + l_knee.y) / 2
        wrist_y = min(r_wrist.y, l_wrist.y)   # lower = higher y value in image
        drop_score = max(0.0, wrist_y - knee_y)

        # Score 2: arm extended downward (wrist well below hip)
        hip_y   = (r_hip.y + l_hip.y) / 2
        extend_score = max(0.0, wrist_y - hip_y - 0.05)

        combined = (drop_score * 0.6 + extend_score * 0.4)

        # Pixel location of the active wrist
        active_wrist = r_wrist if r_wrist.y > l_wrist.y else l_wrist
        wrist_px = (int(active_wrist.x * width), int(active_wrist.y * height))

        return combined, wrist_px

    def _proximity_score(self, person_boxes, litter_boxes, width, height) -> float:
        """
        Simple heuristic: if a litter object is detected within
        a person's lower-body bounding box, score > 0.
        """
        best = 0.0
        for pb in person_boxes:
            px1, py1, px2, py2 = pb["bbox"]
            person_h = py2 - py1
            # Lower third of person box = feet region
            feet_y = py1 + int(person_h * 0.65)
            for lb in litter_boxes:
                lx1, ly1, lx2, ly2 = lb["bbox"]
                lc_x = (lx1 + lx2) / 2
                lc_y = (ly1 + ly2) / 2
                in_x = px1 - 20 <= lc_x <= px2 + 20
                in_y = feet_y <= lc_y <= py2 + 40
                if in_x and in_y:
                    score = lb["confidence"] * pb["confidence"]
                    best  = max(best, score)
        return best

    def _update_tracks(self, tracks, person_boxes, frame_idx, timestamp_s):
        """Very simple track-by-index approach (replace with SORT/DeepSORT in prod)."""
        for i, pb in enumerate(person_boxes):
            if i not in tracks:
                tracks[i] = PersonTrack(
                    track_id=i,
                    first_seen_s=round(timestamp_s, 2),
                    last_seen_s=round(timestamp_s, 2),
                    frame_count=1,
                )
            else:
                tracks[i].last_seen_s = round(timestamp_s, 2)
                tracks[i].frame_count += 1

    def _add_blob(self, acc: np.ndarray, cx: int, cy: int, strength: float = 1.0):
        """Add a Gaussian blob to the heatmap accumulator at (cx, cy)."""
        h, w = acc.shape
        sigma = BLOB_SIGMA
        x0 = max(0, cx - 3 * sigma)
        x1 = min(w, cx + 3 * sigma)
        y0 = max(0, cy - 3 * sigma)
        y1 = min(h, cy + 3 * sigma)
        for y in range(y0, y1, 4):
            for x in range(x0, x1, 4):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                acc[y, x] += strength * np.exp(-d2 / (2 * sigma ** 2))

    def _encode_heatmap(self, acc, width, height, job_id, result_dir) -> "HeatmapData":
        """Normalise, colour-map, and save the heatmap as PNG."""
        if acc.max() > 0:
            norm = (acc / acc.max() * 255).astype(np.uint8)
        else:
            norm = acc.astype(np.uint8)

        # Apply JET colormap and save
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        out_path = Path(result_dir) / f"{job_id}_heatmap.png"
        cv2.imwrite(str(out_path), colour)

        return HeatmapData(
            file_path=str(out_path),
            width=width,
            height=height,
        )

    def _describe_event(self, confidence: float) -> str:
        if confidence >= 0.85:
            return "High-confidence littering action — waste item dropped on ground."
        elif confidence >= 0.65:
            return "Probable littering detected — person bent down and deposited object."
        else:
            return "Possible littering — suspicious arm/hand movement near ground level."

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
