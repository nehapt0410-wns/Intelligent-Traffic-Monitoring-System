# vehicle_monitoring_tracked.py
import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# ----------------- Global variables -----------------
video_path = None          # Path to the selected video
running = False            # Flag to control monitoring thread
monitor_thread = None      # Thread object for monitoring

# Vehicle counters
total_vehicles = 0
car_count = 0
bus_count = 0
truck_count = 0
bike_count = 0

# Tkinter widgets (initialized in build_gui)
file_label = None
status_label = None
counter_label = None
video_label = None

# Set to store tracked vehicle IDs (to avoid double counting)
tracked_ids = set()


# ----------------- Helper functions -----------------

def choose_file():
    """
    Open a file dialog to select a video.
    Update the file_label and status_label accordingly.
    """
    global video_path
    path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
    )
    if path:
        video_path = path
        file_label.config(text=f"Selected: {video_path}")
        status_label.config(text="Status: Ready", foreground="blue")


def start_monitoring():
    """
    Start the monitoring thread.
    Checks if a video is selected and if monitoring is already running.
    """
    global running, monitor_thread

    if video_path is None:
        status_label.config(text="Status: Please select a video file first!", foreground="red")
        return

    if running:
        status_label.config(text="Status: Already running", foreground="orange")
        return

    running = True
    monitor_thread = threading.Thread(target=monitor_traffic, daemon=True)
    monitor_thread.start()
    status_label.config(text="Status: Monitoring started...", foreground="green")


def stop_monitoring():
    """
    Stop the monitoring loop by setting running to False.
    """
    global running
    running = False
    status_label.config(text="Status: Stopped", foreground="red")


def reset_counters():
    """
    Reset all vehicle counters and tracked IDs.
    """
    global total_vehicles, car_count, bus_count, truck_count, bike_count, tracked_ids
    total_vehicles = car_count = bus_count = truck_count = bike_count = 0
    tracked_ids.clear()
    update_counter_label()


def update_counter_label():
    """
    Update the GUI counter label with current counts.
    Thread-safe by using `after` method.
    """
    counter_text = (
        f"Total: {total_vehicles}   "
        f"Cars: {car_count}   "
        f"Buses: {bus_count}   "
        f"Trucks: {truck_count}   "
        f"Bikes: {bike_count}"
    )
    counter_label.config(text=counter_text)


def monitor_traffic():
    """
    Main monitoring loop:
    - Open video
    - Run YOLOv8 tracking on each frame
    - Count vehicles uniquely based on track IDs
    - Update GUI with video and counters
    """
    global running, total_vehicles, car_count, bus_count, truck_count, bike_count, tracked_ids

    # Load YOLOv8 model (nano)
    model = YOLO("yolov8n.pt")

    # Classes considered as vehicles
    vehicle_classes = {"car", "bus", "truck", "motorbike", "motorcycle"}

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_label.config(text="Status: Error opening video!", foreground="red")
        running = False
        return

    # Limit FPS to ~30
    prev_time = 0
    fps_limit = 1 / 30

    while running and cap.isOpened():
        current_time = time.time()
        if current_time - prev_time < fps_limit:
            time.sleep(0.01)
            continue
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Status: Video finished", foreground="blue")
            break

        # Run YOLOv8 with tracking
        # 'tracker="bytetrack.yaml"' uses ByteTrack for ID assignment
        results = model.track(frame, tracker="bytetrack.yaml")[0]

        # Process detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if hasattr(box, 'id') else None

            label = model.names[cls_id].lower()
            if label not in vehicle_classes:
                continue

            # Count only new tracked IDs
            if track_id is not None and track_id not in tracked_ids:
                tracked_ids.add(track_id)
                total_vehicles += 1
                if "car" in label:
                    car_count += 1
                elif "bus" in label:
                    bus_count += 1
                elif "truck" in label:
                    truck_count += 1
                else:
                    bike_count += 1

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw counters on frame
        cv2.putText(
            frame,
            f"Total: {total_vehicles} | Car: {car_count} | Bus: {bus_count} | "
            f"Truck: {truck_count} | Bike: {bike_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Convert to Tkinter format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_pil = img_pil.resize((800, 450))
        imgtk = ImageTk.PhotoImage(image=img_pil)

        # Update GUI (thread-safe)
        video_label.after(0, lambda img=imgtk: video_label.configure(image=img))
        video_label.imgtk = imgtk
        counter_label.after(0, update_counter_label)

    cap.release()
    running = False


# ----------------- GUI Setup -----------------

def build_gui():
    """
    Build the Tkinter GUI with buttons, labels, and video area.
    """
    global file_label, status_label, counter_label, video_label

    root = Tk()
    root.title("Vehicle Traffic Monitoring System")
    root.geometry("1000x700")

    # Top frame for buttons
    top_frame = Frame(root)
    top_frame.pack(side=TOP, fill=X, padx=10, pady=10)

    btn_select = ttk.Button(top_frame, text="Select Video", command=choose_file)
    btn_select.pack(side=LEFT, padx=5)

    btn_start = ttk.Button(top_frame, text="Start Monitoring", command=start_monitoring)
    btn_start.pack(side=LEFT, padx=5)

    btn_stop = ttk.Button(top_frame, text="Stop", command=stop_monitoring)
    btn_stop.pack(side=LEFT, padx=5)

    btn_reset = ttk.Button(top_frame, text="Reset Counters", command=reset_counters)
    btn_reset.pack(side=LEFT, padx=5)

    # Labels under buttons
    file_label = Label(root, text="Selected: None", anchor="w")
    file_label.pack(fill=X, padx=10)

    status_label = Label(root, text="Status: Idle", anchor="w", foreground="blue")
    status_label.pack(fill=X, padx=10)

    counter_label = Label(
        root,
        text="Total: 0   Cars: 0   Buses: 0   Trucks: 0   Bikes: 0",
        font=("Arial", 12, "bold"),
    )
    counter_label.pack(fill=X, padx=10, pady=5)

    # Video frame area
    video_label = Label(root)
    video_label.pack(padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    build_gui()
