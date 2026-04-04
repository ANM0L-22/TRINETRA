"""
Trinetra — Simulator (video-aware wrapper)
When a video is loaded, analytics pull from video_engine.
This module provides fallback data and historical generation.
"""

import random
import math
import time
from datetime import datetime, timedelta
from collections import deque

VEHICLE_CLASSES = ["car", "bus", "bike", "truck", "auto_rickshaw", "pedestrian"]
VIOLATION_TYPES = ["Helmet Violation", "Wrong-side Driving",
                   "Mobile Usage", "Tampered Plate", "No Seatbelt"]
INDIAN_STATES   = ["MH", "DL", "RJ", "KA", "TN", "UP", "GJ", "WB"]
INTERSECTIONS   = ["NH-48 · Sector 7","MG Road · Junction 3",
                   "Ring Road · Gate 12","Outer Ring · Node 6","City Center · Hub 1"]


def _gen_plate():
    s = random.choice(INDIAN_STATES)
    d = str(random.randint(1,99)).zfill(2)
    let = "".join(random.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=2))
    num = str(random.randint(1000,9999))
    return f"{s}{d}{let}{num}"


def get_frame_data(frame_id: int, density_trend: float = 0.4) -> dict:
    """Fallback frame data when no video is loaded."""
    rng = random.Random(frame_id + int(time.time() * 0.1))
    n   = rng.randint(3, 12)
    counts = {c: 0 for c in VEHICLE_CLASSES}
    vehicles = []
    for i in range(n):
        cls = rng.choices(VEHICLE_CLASSES, weights=[35,8,28,10,12,7])[0]
        counts[cls] += 1
        vehicles.append({"id":i+1,"class":cls,"conf":round(rng.uniform(.72,.98),2),
                          "track_id":rng.randint(100,999),"speed_kmh":round(rng.uniform(8,62),1)})
    noise = math.sin(frame_id * 0.05) * 0.12 + rng.uniform(-0.08,0.08)
    density = float(max(0.05, min(0.95, density_trend + noise)))
    congestion = "LOW" if density<0.25 else "MEDIUM" if density<0.5 else "HIGH" if density<0.75 else "CRITICAL"
    viols = []
    for v in vehicles:
        if rng.random() < 0.08:
            viols.append({"type":rng.choice(VIOLATION_TYPES),"plate":_gen_plate(),
                           "vehicle":v["class"],"conf":round(rng.uniform(.6,.95),2)})
    return {"frame_id":frame_id,"n_vehicles":n,"counts":counts,"vehicles":vehicles,
            "violations":viols,"density":round(density,4),"congestion":congestion,
            "fps":round(rng.uniform(22,32),1),"proc_ms":round(rng.uniform(18,48),1),
            "decisions":[],"plates":[],"frame_b64":""}


def get_historical_data(hours: int = 6) -> list:
    records = []
    base = datetime.now() - timedelta(hours=hours)
    for i in range(hours * 60):
        t    = base + timedelta(minutes=i)
        hour = t.hour
        rush = 1.0 if 8<=hour<=10 or 17<=hour<=20 else 0.3
        noise= random.uniform(-0.1,0.1)
        count= int((rush+noise)*18+random.randint(2,6))
        records.append({
            "time":t.strftime("%H:%M"),"timestamp":t,
            "vehicles":count,"density":round(min(rush+noise+random.uniform(0,.2),1.),3),
            "violations":random.randint(0,3) if rush>0.5 else random.randint(0,1),
            "speed_avg":round(random.uniform(15,55)*(1-rush*.5),1),
            "cars":int(count*.35),"bikes":int(count*.28),"buses":int(count*.08),
            "trucks":int(count*.10),"autos":int(count*.12),
            "congestion": "HIGH" if rush>0.7 else "MEDIUM" if rush>0.4 else "LOW",
        })
    return records


def get_violation_history(n: int = 80) -> list:
    records = []
    base = datetime.now() - timedelta(hours=8)
    for i in range(n):
        t = base + timedelta(minutes=random.randint(0,480))
        records.append({
            "id":f"TRN-{i+1:05d}","type":random.choice(VIOLATION_TYPES),
            "plate":_gen_plate(),"timestamp":t.strftime("%Y-%m-%d %H:%M:%S"),
            "location":random.choice(INTERSECTIONS),
            "confidence":round(random.uniform(.62,.97),3),
            "severity":random.choice(["critical","high","medium"]),
            "fine_inr":random.choice([500,1000,1500,2000]),
            "status":random.choice(["Pending","Issued","Paid"]),
        })
    records.sort(key=lambda x:x["timestamp"],reverse=True)
    return records
