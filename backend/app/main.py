import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, uuid, json
from datetime import datetime
from .analysis import UPLOADS, analyze_video, FRAMES
from .models import AnalysisResult

BASE_DIR = Path(__file__).resolve().parents[1]
EVENTS_STORE = BASE_DIR / "events_store.json"
if not EVENTS_STORE.exists():
    EVENTS_STORE.write_text("[]")

app = FastAPI(title="HomeCareAI Demo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve saved frames and static
static_root = BASE_DIR / "static"
static_root.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_root)), name="static")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # save upload
    vid_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or ".mp4"
    save_path = UPLOADS / f"{vid_id}{suffix}"
    with save_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"video_id": vid_id, "filename": file.filename, "path": str(save_path)}

@app.post("/analyze/{video_id}")
async def analyze(video_id: str):
    # find file in uploads (try common extensions)
    p = None
    for ext in [".mp4", ".mov", ".mkv", ".avi"]:
        cand = UPLOADS / f"{video_id}{ext}"
        if cand.exists():
            p = cand
            break
    if p is None:
        raise HTTPException(status_code=404, detail="video not found")
    # call analyzer (your comprehensive analysis.py)
    res = analyze_video(str(p), video_id, fps_sample=1.0)

    # store events into events_store.json (assign ids)
    events = json.loads(EVENTS_STORE.read_text())
    for e in res.get("events", []):
        eid = f"evt-{len(events)+1:03d}"
        e_record = {"id": eid, **e}
        events.append(e_record)
    EVENTS_STORE.write_text(json.dumps(events, indent=2))

    # return analysis result (including representative frame path, notifications)
    # optionally include a human_summary key if your analysis provides it
    return res

@app.get("/events")
async def list_events():
    return json.loads(EVENTS_STORE.read_text())

@app.get("/events/{eid}")
async def get_event(eid: str):
    events = json.loads(EVENTS_STORE.read_text())
    for e in events:
        if e.get("id") == eid:
            return e
    raise HTTPException(status_code=404, detail="event not found")

@app.post("/events/{eid}/confirm")
async def confirm_event(eid: str):
    """
    Mark an event as confirmed (useful for requires_confirmation semantics).
    Adds a confirmation notification to the stored event.
    """
    events = json.loads(EVENTS_STORE.read_text())
    found = False
    for e in events:
        if e.get("id") == eid:
            found = True
            # mark confirmed (remove requires_confirmation or set flag)
            e["requires_confirmation"] = False
            note = {"channel":"sms","message":f"Confirmed event {eid} - emergency actions allowed","time": datetime.utcnow().isoformat()+"Z"}
            e.setdefault("notifications", []).append(note)
            break
    if not found:
        raise HTTPException(status_code=404, detail="event not found")
    EVENTS_STORE.write_text(json.dumps(events, indent=2))
    return {"status":"ok", "event": e}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=4078, reload=True)
