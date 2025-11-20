# Intelligent-Traffic-Monitoring-System

A real-time vehicle detection and counting system using **YOLOv8**, **OpenCV**, and a **Tkinter GUI**.

This application allows users to load a video file, run real-time vehicle detection, count different types of vehicles, and visualize the processed frames directly in a Tkinter window.

---

## 📌 Features
- Load video files (`.mp4`, `.avi`, `.mov`, `.mkv`, etc.)
- Real-time YOLOv8 vehicle detection
- Counts:
  - Cars  
  - Buses  
  - Trucks  
  - Motorbikes  
  - Total vehicles
- Start / Stop monitoring
- Reset counters anytime
- Live video preview on GUI
- Multithreaded processing (GUI never freezes)

## 🔧 Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python pillow numpy

vehicle_monitoring.py
YOLOv8n model will download automatically on first run
python vehicle_monitoring.py

🖥️ How It Works
1. User selects a video file

Using Tkinter file dialog.

2. User clicks Start Monitoring

A background thread begins reading frames via OpenCV.

3. YOLOv8 runs on each frame

Identifies vehicles:

car

bus

truck

motorbike/motorcycle

4. Detected vehicles are counted

Counters update in real time.

5. Bounding boxes are drawn

Frame is converted to RGB and displayed in Tkinter window.
6. User may stop or reset counters anytime
🚗 Vehicle Classes Detected

The YOLO model detects many classes, but this program counts only:

YOLO Label	Counted As
car	Car
bus	Bus
truck	Truck
motorbike	Bike
motorcycle	Bike

