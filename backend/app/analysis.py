import cv2
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOADS = BASE_DIR / "uploads"
FRAMES = BASE_DIR / "static" / "frames"
EVENTS_JSON = BASE_DIR / "events_store.json"
FRAMES.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
if not EVENTS_JSON.exists():
    EVENTS_JSON.write_text("[]")

# load face detector (OpenCV Haar cascade shipped with opencv-python)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_frames(video_path: str, fps_sample: float = 1.0) -> List[Dict]:
    """
    Extract frames at approx fps_sample frames per second.
    Returns list of dicts: { 't': seconds, 'frame': numpy array }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / fps_sample)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append({"t": t, "frame": frame.copy()})
        idx += 1
    cap.release()
    return frames

def detect_faces(frame: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    out = []
    for (x,y,w,h) in faces:
        out.append({"bbox":[int(x),int(y),int(w),int(h)], "area": int(w*h)})
    return out

def detect_fire_smoke(frame: np.ndarray) -> Dict:
    """
    Simple color-based heuristic:
    - Convert to HSV and look for orange/red/yellowish regions for flames
    - Look for greyish low-sat regions for smoke
    Returns dict with scores 0..1
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    # flame/orange/red mask
    lower1 = np.array([0, 80, 120])
    upper1 = np.array([20, 255, 255])
    lower2 = np.array([160, 80, 120])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    flame_mask = cv2.bitwise_or(mask1, mask2)
    flame_frac = (np.count_nonzero(flame_mask) / flame_mask.size) if flame_mask.size else 0.0
    # smoke: low saturation, mid-high brightness
    smoke_mask = cv2.inRange(hsv, np.array([0,0,120]), np.array([179,40,255]))
    smoke_frac = (np.count_nonzero(smoke_mask) / smoke_mask.size) if smoke_mask.size else 0.0
    return {"flame_score": float(min(1.0, flame_frac*5)), "smoke_score": float(min(1.0, smoke_frac*5))}

def detect_motion(frames: List[Dict]) -> List[Dict]:
    """
    Very simple motion detector via frame differences. Returns timepoints with motion score.
    """
    motions = []
    prev = None
    for item in frames:
        f = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2GRAY)
        f = cv2.GaussianBlur(f,(5,5),0)
        if prev is None:
            prev = f
            continue
        diff = cv2.absdiff(prev, f)
        _,th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_frac = np.count_nonzero(th) / th.size
        motions.append({"t": item["t"], "motion_score": float(motion_frac)})
        prev = f
    return motions

def detect_fall(frames: List[Dict]) -> Dict:
    """
    Heuristic: detect a person bounding box (via face or rough person detection).
    If a person's bounding box centroid drops in vertical position across frames and then remains low -> fall.
    Here we use face detector as proxy (not perfect).
    """
    centers = []
    for item in frames:
        faces = detect_faces(item["frame"])
        if faces:
            # pick largest face
            f = max(faces, key=lambda x: x["area"])
            x,y,w,h = f["bbox"]
            centers.append({"t": item["t"], "cy": float(y + h/2)})
    if len(centers) < 2:
        return {"detected": False, "confidence": 0.0}
    # check for downward shift > threshold (pixel space)
    cy_values = [c["cy"] for c in centers]
    drop = cy_values[-1] - cy_values[0]
    if drop > 50:  # heuristic threshold
        return {"detected": True, "confidence": min(1.0, drop/200.0)}
    return {"detected": False, "confidence": 0.0}

def analyze_video(video_path: str, video_id: str, fps_sample: float = 1.0) -> Dict:
    frames = extract_frames(video_path, fps_sample=fps_sample)
    if not frames:
        return {"video_id": video_id, "events": [], "representative_frame": None, "notifications": []}
    # run detectors on sampled frames
    events = []
    notifications = []
    rep_frame_path = None

    # face & fire/smoke detection per frame
    flame_hits = []
    smoke_hits = []
    face_hits = []
    for f in frames:
        ff = f["frame"]
        fs = detect_fire_smoke(ff)
        faces = detect_faces(ff)
        if fs["flame_score"] > 0.15 or fs["smoke_score"] > 0.2:
            flame_hits.append({"t": f["t"], **fs})
        if faces:
            face_hits.append({"t": f["t"], "faces": faces})
    # motion
    motions = detect_motion(frames)
    high_motion = [m for m in motions if m["motion_score"] > 0.02]

    # fall: use faces trajectory heuristic
    fall_res = detect_fall(frames)

    # rules to assemble events (simple thresholds)
    if fall_res["detected"] and fall_res["confidence"] >= 0.4:
        t = frames[0]["t"]
        events.append({"type":"fall","timestamp":datetime.utcnow().isoformat()+"Z","room":"Living Room","message":"Fall detected (heuristic)","severity":"critical","confidence":round(fall_res["confidence"],2)})
        notifications.append({"channel":"sms","message":"Fall detected — notify caregiver","time":datetime.utcnow().isoformat()+"Z"})
    if flame_hits:
        events.append({"type":"fire","timestamp":datetime.utcnow().isoformat()+"Z","room":"Kitchen","message":"Fire/smoke detected (color heuristic)","severity":"critical","confidence":round(max(h.get("flame_score",0) for h in flame_hits),2)})
        notifications.append({"channel":"sms","message":"Fire detected — emergency protocol","time":datetime.utcnow().isoformat()+"Z"})
    if face_hits and not flame_hits:
        # if faces found but not matching known list, flag as visitor
        events.append({"type":"visitor","timestamp":datetime.utcnow().isoformat()+"Z","room":"Front Door","message":"Visitor/face detected (needs verification)","severity":"high","confidence":round( min(1.0, 0.5+0.1*len(face_hits)),2) })
        notifications.append({"channel":"push","message":"Unrecognized visitor — please review","time":datetime.utcnow().isoformat()+"Z"})
    if high_motion and not (flame_hits or fall_res["detected"]):
        events.append({"type":"motion","timestamp":datetime.utcnow().isoformat()+"Z","room":"Living Room","message":"Motion detected","severity":"info","confidence":round(max(m["motion_score"] for m in high_motion),2)})

    # save a representative frame (first frame with a detection)
    rep = None
    for f in frames:
        ff = f["frame"]
        fs = detect_fire_smoke(ff)
        faces = detect_faces(ff)
        if fs["flame_score"] > 0.15 or faces or (len(frames)>1 and detect_motion([f])!=[]):
            rep = ff
            break
    if rep is None:
        rep = frames[len(frames)//2]["frame"]
    # write rep to disk
    rep_fname = f"{video_id}_rep.jpg"
    rep_path = FRAMES / rep_fname
    cv2.imwrite(str(rep_path), rep)
    rep_frame_path = f"/static/frames/{rep_fname}"

    return {"video_id": video_id, "events": events, "representative_frame": rep_frame_path, "notifications": notifications}
