"""
schemas.py — Pydantic data models for the Littering Detection API
"""

from pydantic import BaseModel, Field
from typing import Optional


class LitteringEvent(BaseModel):
    """A single detected littering incident."""
    timestamp_s:   float             = Field(..., description="Time offset in seconds")
    timestamp_str: str               = Field(..., description="MM:SS formatted time")
    confidence:    float             = Field(..., description="Detection confidence 0–1")
    frame_number:  int               = Field(..., description="Source frame index")
    location_xy:   Optional[list]    = Field(None, description="[x, y] pixel location")
    description:   str               = Field(..., description="Human-readable description")


class PersonTrack(BaseModel):
    """Simple person track across frames."""
    track_id:     int   = Field(..., description="Track identifier")
    first_seen_s: float = Field(..., description="Timestamp when first detected")
    last_seen_s:  float = Field(..., description="Timestamp when last detected")
    frame_count:  int   = Field(..., description="Number of frames this person appeared in")


class HeatmapData(BaseModel):
    """Heatmap output file info."""
    file_path: str = Field(..., description="Server-side path to heatmap PNG")
    width:     int = Field(..., description="Original video width")
    height:    int = Field(..., description="Original video height")


class AnalysisResult(BaseModel):
    """Full analysis result returned after processing."""
    job_id:            str
    verdict:           str              = Field(..., description="LITTERING_DETECTED or CLEAN")
    littering_detected: bool
    event_count:       int
    events:            list[LitteringEvent]
    frames_analysed:   int
    total_frames:      int
    persons_detected:  int
    duration_s:        float
    avg_confidence:    float
    fps:               float
    resolution:        str
    person_tracks:     list[PersonTrack] = []
    heatmap:           Optional[HeatmapData] = None


class JobStatus(BaseModel):
    """Job polling response."""
    job_id:   str
    status:   str              = Field(..., description="queued | processing | done | error")
    progress: int              = Field(..., description="0–100 percent complete")
    result:   Optional[dict]   = None
