# 🚀 QUICK START GUIDE — TRINETRA

**Get Started in 3 Minutes**

---

## ✅ Installation

### 1. Install Dependencies (First Time Only)
```bash
cd c:\Users\goswa\Downloads\trinetra\ final\ dashboard\trinetra
pip install -r requirements.txt
```

> ⏱️ Takes ~2-5 minutes (downloads YOLOv8 model ~49MB)

### 2. Verify System
```bash
python verify_system.py
```

Expected output:
```
✅ ALL CHECKS PASSED (7/7)
🚀 Ready to start Trinetra!
```

---

## 🎬 Running Trinetra

### Start Dashboard
```bash
streamlit run app.py
```

Open browser: **`http://localhost:8501`**

### Login Credentials
- **Username:** `admin`
- **Password:** `trinetra@2024`

---

## 📹 How to Use

### Step 1: Upload Video
1. Navigate to **LAYER 1 · VIDEO UPLOAD & ANALYSIS**
2. Click **📂 Upload Traffic Video**
3. Select your traffic video (MP4, AVI, MOV, MKV)
4. Watch status: 📤 UPLOADING → 🔄 ANALYZING → ✅ COMPLETE

### Step 2: View Detections
1. Use **frame slider** to navigate through video
2. Click **▶ Analyse Frame** to see detections
   - Green boxes: Cars
   - Blue boxes: Buses
   - Pink boxes: Bikes
   - Orange boxes: Trucks
   - Light green boxes: Auto-rickshaws

### Step 3: Check Violations
1. Look at **Layer 1 Right** section for violation alerts
2. Violations show: Vehicle type, Plate number, Violation type, Confidence
3. Examples:
   - 🪖 No Helmet (on bikes)
   - ↩️ Wrong-side Driving
   - 🚫 Tampered Plates
   - 🚦 Red Light Violations

### Step 4: View Analytics
1. Scroll to **LAYER 3 · COMPREHENSIVE ANALYTICS**
2. See charts:
   - 📊 Vehicle Count Over Time
   - 📈 Congestion Score Timeline
   - 🎯 Vehicle Class Distribution
   - ⚠️ Violation Statistics

---

## 🎮 Control Buttons

| Button | What It Does |
|--------|-------------|
| **▶ Analyse Frame** | Run detection on current frame |
| **⏭ Next Frame** | Jump to next frame |
| **⏮ Prev Frame** | Jump to previous frame |
| **🔄 Refresh** | Reset analysis, clear state |

---

## 🎯 What You'll See

### Real Detections
- ✅ Actual vehicle bounding boxes (from YOLOv8)
- ✅ Actual number plates (from PaddleOCR)
- ✅ Actual violations (from detection engine)
- ✅ Real confidence scores (not fake data)

### Analytics Data
- ✅ Vehicle count matches detections
- ✅ Congestion calculated from frame content
- ✅ Violations from actual detection
- ✅ All data from single uploaded video

### Status Updates
- 📁 IDLE — Ready for upload
- 📤 UPLOADING — Saving video
- 🔄 ANALYZING — Running detection
- ✅ COMPLETE — Analysis done
- ❌ ERROR — Something failed

---

## 🎨 Understand the UI

### LAYER 1 (Top Section)
- **Left:** Video upload, status display
- **Center:** Video player + frame analyzer
- **Right:** Traffic map, congestion indicator

### LAYER 2 (Middle Section)
- **Left:** Detected vehicles list
- **Center:** Frame with detection boxes
- **Right:** Real-time violations alert

### LAYER 3 (Bottom Section)
- **Charts:** Vehicle counts, congestion, class distribution
- **Real data** from analyzed video samples

---

## 📊 Understanding Results

### Vehicle Detection Box
```
┌─────────────────────────────────┐
│ Confidence: 0.95                │
│ Class: car                      │
│ Plate: MH02AB1234              │
└─────────────────────────────────┘
```

### Violation Card
```
🪖 NO HELMET
Vehicle: bike | Plate: MH02XY5678
Confidence: 0.92
```

### Traffic Map
```
Vehicle Count: 8
Congestion Level: MEDIUM (56%)
Status: 🟡 Yellow (moderate traffic)
```

---

## 💡 Tips & Tricks

### For Better Detection
1. Use clear, well-lit videos
2. Keep vehicles in frame for >0.5 seconds
3. Use 720p+ resolution videos
4. Ensure plates are visible

### For Faster Processing
1. Use YOLOv8n model (nano, faster)
2. Reduce frame resolution
3. Use GPU if available

### For Better Accuracy
1. Use YOLOv8l model (larger, more accurate)
2. Ensure good lighting in video
3. Use high-resolution footage

---

## 🚨 Common Issues & Solutions

### Issue: "Real engine not available"
**Solution:** Install YOLOv8
```bash
pip install ultralytics>=8.2.0
```

### Issue: "No violations detected"
This is normal! The system only detects actual violations, not random ones.

### Issue: Slow processing?
**Solution:** Check:
- CPU load (use GPU if available)
- Video resolution (reduce if too large)
- Model size (use yolov8n for speed)

### Issue: Plates not recognized?
**Solution:** 
- Use clearer videos
- Ensure plates are visible
- Check video resolution

---

## 📖 Learn More

1. **README_PRODUCTION.md** — Full documentation
2. **FIXES_IMPLEMENTED.md** — Technical details
3. **DEPLOYMENT_SUMMARY.md** — System overview
4. **verify_system.py** — Run verification

---

## 🎓 Example Workflow

**Test Video Processing:**

1. **Upload a video** (2-5 minute CCTV footage)
2. **Wait for analysis** (status shows progress)
3. **Navigate frames** using slider
4. **Click Analyse Frame** on interesting scenes
5. **View detections** in center panel
6. **Check violations** on right panel
7. **Scroll down** for analytics charts

**Expected Results:**
- ✅ Vehicle boxes in video
- ✅ Real plate numbers shown
- ✅ Violations detected if applicable
- ✅ Charts show actual data

---

## 🚀 Advanced Features

### Custom Zones
Edit `src/violation/real_detector.py` to define:
- Red light zones
- Wrong-side lanes
- Speed limit areas

### Custom Models
Train YOLOv8 on Indian traffic:
```bash
python scripts/train.py --data configs/dataset.yaml --epochs 100
```

### Database Integration
Connect to your database by modifying:
- `app/pages/violations.py`
- Add database write functions

---

## 📞 Support

1. Check documentation files
2. Run `python verify_system.py`
3. Review logs in `results/logs/`
4. Check GitHub issues

---

## ✨ What's Different from Before?

| Old System | New System |
|-----------|-----------|
| Random data | Real detections from YOLOv8 |
| Fake plates | Real plates from PaddleOCR |
| Static UI | Dynamic interactive UI |
| Dummy buttons | Fully functional buttons |
| No feedback | Clear status updates |
| Hardcoded values | Real calculated metrics |

---

## 🎉 You're Ready!

```bash
# Start the system
streamlit run app.py

# Upload a video
# Press "Analyse Frame"
# Watch detections appear
# View real analytics
```

**Enjoy using Trinetra!** 🔱

---

*Questions? See full documentation in README_PRODUCTION.md*
