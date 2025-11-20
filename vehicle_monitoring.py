# vehicle_monitoring.py
import cv2
import threading
import time
#user_input = input(">> ")
import numpy as np
from ultralytics import YOLO

from tkinter import *
from tkinter import filedialog, ttk

from PIL import Image, ImageTk
# ----------------- Global variables -----------------
video_path = None
running = False
monitor_thread = None

# vehicle counters
total_vehicles = 0
car_count = 0
bus_count = 0
truck_count = 0
bike_count = 0

# Tkinter widgets (will be created later)
file_label = None
status_label = None
counter_label = None
video_label = None

# ----------------- Helper functions -----------------
def choose_file():
    """Open file dialog and select video file"""
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
    """Start monitoring thread"""
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
    """Stop monitoring loop"""
    global running
    running = False
    status_label.config(text="Status: Stopped", foreground="red")


def reset_counters():
    """Reset all vehicle counters"""
    global total_vehicles, car_count, bus_count, truck_count, bike_count
    total_vehicles = car_count = bus_count = truck_count = bike_count = 0
    update_counter_label()


def update_counter_label():
    """Refresh the counter label text in the GUI"""
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
    - Read frames from video
    - Run YOLOv8 detection
    - Count vehicles by type
    - Show frames in Tkinter window
    """
    global running, total_vehicles, car_count, bus_count, truck_count, bike_count

    # Load YOLO model (downloads yolov8n.pt automatically if not present)
    model = YOLO("yolov8n.pt")

    # Classes we consider as vehicles
    vehicle_classes = {"car", "bus", "truck", "motorbike", "motorcycle"}

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        status_label.config(text="Status: Error opening video!", foreground="red")
        running = False
        return

    # Simple FPS limit to avoid overloading CPU
    prev_time = 0
    fps_limit = 1 / 30  # ~30 FPS

    while running and cap.isOpened():
        current_time = time.time()
        if current_time - prev_time < fps_limit:
            continue
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Status: Video finished", foreground="blue")
            break

        # Run YOLOv8 on frame
        results = model(frame, verbose=False)[0]

        # For each detection, draw box and update counters
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            if label.lower() not in vehicle_classes:
                continue

            # Count as soon as we see it (simple counting, no tracking)
            total_vehicles += 1
            if "car" in label.lower():
                car_count += 1
            elif "bus" in label.lower():
                bus_count += 1
            elif "truck" in label.lower():
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

        # Draw a small info bar
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

        # Convert BGR -> RGB for Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        # Resize to fit the Tkinter label (optional)
        img_pil = img_pil.resize((800, 450))  # adjust size as you like

        imgtk = ImageTk.PhotoImage(image=img_pil)

        # Update GUI frame
        video_label.imgtk = imgtk  # keep reference!
        video_label.configure(image=imgtk)

        # Update counters in GUI
        update_counter_label()

    cap.release()
    running = False


# ----------------- Build the GUI -----------------
def build_gui():
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