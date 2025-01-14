import time
import psutil
import csv
import threading
import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np
import torch

# Performance Logger Class
class PerformanceLogger:
    """
    A lightweight logger that records the current process's CPU and memory
    usage at regular intervals.
    Results are saved to a CSV file.
    """
    def __init__(self, log_filename="performance_log_dl.csv", interval=1.0):
        """
        :param log_filename: Path for the output CSV file.
        :param interval: Time (in seconds) between each sampling of
        CPU/memory usage.
        """
        self.log_filename = log_filename
        self.interval = interval
        self._stop_flag = False
        self._thread = None
        
        # Track this process specifically
        self._process = psutil.Process()
        
    def start(self):
        """Start logging in a background thread."""
        self._stop_flag = False
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Signal the logger to stop and wait for the thread to finish."""
        self._stop_flag = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()   
            
    def _log_loop(self):
        """Internal loop that writes CPU & memory usage to CSV at each interval."""
        with open(self.log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header row
            writer.writerow(["timestamp", "cpu_percent", "memory_mb"])
            # Prime the CPU usage to avoid inaccuracies
            self._process.cpu_percent(interval=None)
            
            while not self._stop_flag:
                timestamp = time.time()
                cpu_percent = self._process.cpu_percent(interval=None)  # CPU % since last call
                mem_info = self._process.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)
                writer.writerow([timestamp, cpu_percent, memory_mb])
                print(f"Logged: {timestamp}, {cpu_percent}%, {memory_mb} MB")  # Print after each log entry
                time.sleep(self.interval)

# Load the Hugging Face transformer model
device = 'cpu'
model_name = "dima806/closed_eyes_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
processor = AutoImageProcessor.from_pretrained(model_name)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat')

# Real-world width of a human face (average ~160 mm)
KNOWN_FACE_WIDTH = 160.0  # in mm
FOCAL_LENGTH = 650  # Focal length in pixels (adjust as needed)

# Eye landmarks indices
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Average frown distance thresholds
AVG_FROWN_DISTANCE_22_28 = 20  # Threshold for frown between 22-28
AVG_FROWN_DISTANCE_22_21 = 20  # Threshold for frown between 22-21
AVG_FROWN_DISTANCE_21_22 = 12  # Threshold for frown between 21-22

# Blink detection variables
previous_eye_state = {"Left Eye": "Open", "Right Eye": "Open"}
blink_cycle_detected = {"Left Eye": False, "Right Eye": False}
blink_count = 0

# Frown detection variables
frown_count = 0

# Distance counter variables
distance_less_than_threshold_count = 0
DISTANCE_THRESHOLD = 508.0  # in mm

# Initialize transition tracker for close distance
previous_distance_above_threshold = True  # Assume initial state is above threshold
transition_to_close_distance_count = 0

def mark_eye_landmark(eye, img, color=(0, 255, 0), scale_factor=1):
    """
    Mark the eye landmarks on the image and return the scaled eye region.
    """
    top = int(max(0, eye[:, 1].min() - scale_factor * (eye[:, 1].max() - eye[:, 1].min())))
    bottom = int(min(img.shape[0], eye[:, 1].max() + scale_factor * (eye[:, 1].max() - eye[:, 1].min())))
    left = int(max(0, eye[:, 0].min() - scale_factor * (eye[:, 0].max() - eye[:, 0].min())))
    right = int(min(img.shape[1], eye[:, 0].max() + scale_factor * (eye[:, 0].max() - eye[:, 0].min())))

    return top, bottom, left, right

# Initialize and start the performance logger
logger = PerformanceLogger(log_filename="performance_log.csv", interval=1.0)
logger.start()

# Video writer setup
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cam.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

frame_count = 0
fps_start_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=640)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    img = frame.copy()

    text_lines = []  # List to hold text lines for display

    for face in faces:
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Estimate distance of the face from the camera
        face_distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / w
        text_lines.append(f"Face Distance: {face_distance:.2f} mm")

        # Count instances where distance is less than threshold
        if face_distance < DISTANCE_THRESHOLD:
            distance_less_than_threshold_count += 1

        # Check for transition from above threshold to below threshold
        if previous_distance_above_threshold and face_distance < DISTANCE_THRESHOLD:
            transition_to_close_distance_count += 1

        # Update the previous state
        previous_distance_above_threshold = face_distance >= DISTANCE_THRESHOLD

        shape = landmark_predict(img_gray, face)
        shape = face_utils.shape_to_np(shape)

        # Frown detection
        point_22 = shape[22]
        point_28 = shape[28]
        point_21 = shape[21]

        dist_22_28 = dist.euclidean(point_22, point_28)
        dist_21_22 = dist.euclidean(point_21, point_22)
        dist_21_28 = dist.euclidean(point_21, point_28)

        if dist_22_28 < AVG_FROWN_DISTANCE_22_28 or dist_21_22 < AVG_FROWN_DISTANCE_21_22 or dist_22_28 < AVG_FROWN_DISTANCE_22_21:
            text_lines.append("Frown Detected")
            frown_count += 1

        text_lines.append(f"Dist 21-22: {dist_21_22:.2f}")
        text_lines.append(f"Dist 22-28: {dist_22_28:.2f}")
        text_lines.append(f"Dist 21-28: {dist_21_28:.2f}")

        # Blink detection
        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        for eye_img, eye_label in zip([lefteye, righteye], ["Left Eye", "Right Eye"]):
            top, bottom, left, right = mark_eye_landmark(eye_img, img)
            eye_region = img[top:bottom, left:right]

            if eye_region.size > 0:
                eye_region_pil = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
                inputs = processor(images=eye_region_pil, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    state = model.config.id2label[predicted_class_idx]

                if previous_eye_state["Left Eye"] == "openEye" and state == "closeEye" and eye_label == "Left Eye":
                    blink_cycle_detected["Left Eye"] = True
                elif previous_eye_state["Right Eye"] == "openEye" and state == "closeEye" and eye_label == "Right Eye":
                    blink_cycle_detected["Right Eye"] = True

                if (blink_cycle_detected["Left Eye"] and state == "openEye" and eye_label == "Left Eye") or \
                   (blink_cycle_detected["Right Eye"] and state == "openEye" and eye_label == "Right Eye"):

                    if blink_cycle_detected["Left Eye"] and blink_cycle_detected["Right Eye"]:
                        blink_count += 1
                        blink_cycle_detected["Left Eye"] = False
                        blink_cycle_detected["Right Eye"] = False

                previous_eye_state[eye_label] = state
                text_lines.append(f"{eye_label}: {state}")

    text_lines.append(f"Blinks: {blink_count}")
    text_lines.append(f"Frowns: {frown_count}")
    # text_lines.append(f"Close Distance Count (<{DISTANCE_THRESHOLD:.0f} mm): {distance_less_than_threshold_count}")
    text_lines.append(f"Transitions to Close Distance: {transition_to_close_distance_count}")

    # Calculate FPS
    frame_count += 1
    fps_end_time = time.time()
    fps = frame_count / (fps_end_time - fps_start_time)
    text_lines.append(f"FPS: {fps:.2f}")

    # Display all text lines in the top-left corner
    line_height = 20
    start_x, start_y = 10, 20
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (start_x, start_y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Monitoring", img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the logger and release the webcam
logger.stop()
cam.release()
out.release()
cv2.destroyAllWindows()
