from pydantic import BaseModel
from typing import List, Optional, Dict


class Event(BaseModel):
id: str
type: str
timestamp: str
room: str
message: str
severity: str
confidence: float


class AnalysisResponse(BaseModel):
video_id: str
events: List[Event]
representative_frame: Optional[str]
notifications: List[Dict]