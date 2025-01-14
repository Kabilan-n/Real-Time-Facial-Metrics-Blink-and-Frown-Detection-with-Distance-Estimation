import time
import psutil
import csv
import threading
import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils

class PerformanceLogger:
    """
    A lightweight logger that records the current process's CPU and memory
    usage at regular intervals.
    Results are saved to a CSV file.
    """
    def __init__(self, log_filename="performance_log_geo.csv", interval=1.0):
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

# Initialize webcam
cam = cv2.VideoCapture(0)

# Get video properties and set up video writer
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)
output_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Initialize Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat')

# Real-world width of a human face (average ~160 mm)
KNOWN_FACE_WIDTH = 160.0  # in mm
FOCAL_LENGTH = 650  # Focal length in pixels (adjust as needed)
DISTANCE_THRESHOLD = 508  # Distance threshold in mm for transition

# Eye landmarks indices
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Average frown distance thresholds
AVG_FROWN_DISTANCE_22_28 = 24  # Threshold for frown between 22-28
AVG_FROWN_DISTANCE_22_21 = 24  # Threshold for frown between 22-21
AVG_FROWN_DISTANCE_21_22 = 13  # Threshold for frown between 21-22

# Blink detection variables
blink_thresh = 0.3
succ_frame = 2
count_frame = 0
blink_count = 0

# Frown detection variables
frown_count = 0
previous_frown_state = False

# Distance transition variables
distance_less_than_threshold_count = 0
transition_to_close_distance_count = 0
previous_distance_above_threshold = True

def calculate_EAR(eye):
    """Calculate the Eye Aspect Ratio (EAR) for detecting blinks."""
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / x1
    return EAR

# Mark the eye landmarks on the image
def mark_eyeLandmark(img, eyes):
    for eye in eyes:
        pt1, pt2 = (eye[1], eye[5])
        pt3, pt4 = (eye[0], eye[3])
    return img

# Initialize and start the performance logger
logger = PerformanceLogger(log_filename="performance_log.csv", interval=1.0)
logger.start()

# Start time for FPS calculation
start_time = time.time()
frame_count = 0

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

        current_frown_state = dist_22_28 < AVG_FROWN_DISTANCE_22_28 or dist_21_22 < AVG_FROWN_DISTANCE_21_22 or dist_22_28 < AVG_FROWN_DISTANCE_22_21
        if current_frown_state and not previous_frown_state:
            frown_count += 1
        previous_frown_state = current_frown_state

        if current_frown_state:
            text_lines.append("Frown Detected")

        text_lines.append(f"Dist 21-22: {dist_21_22:.2f}")
        text_lines.append(f"Dist 22-28: {dist_22_28:.2f}")
        text_lines.append(f"Dist 21-28: {dist_21_28:.2f}")

        # Blink detection
        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        # Calculate EAR for both eyes
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)
        img = mark_eyeLandmark(img, [lefteye, righteye])

        # Average EAR of both eyes
        avg = (left_EAR + right_EAR) / 2
        if avg < blink_thresh:
            count_frame += 1
        else:
            if count_frame >= succ_frame:
                blink_count += 1
            count_frame = 0

    # Display blink, frown, and distance transition counts
    text_lines.append(f'Blinks: {blink_count}')
    text_lines.append(f'Frowns: {frown_count}')
    text_lines.append(f'Transitions to Close Distance: {transition_to_close_distance_count}')

    # Format and display text lines in a corner
    line_height = 20  # Line spacing
    start_x, start_y = 10, 30  # Starting coordinates
    box_padding = 10  # Padding around the text box
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 255, 0)

    # Calculate the width and height of the text box
    max_line_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][0] for line in text_lines], default=0)
    box_height = line_height * len(text_lines) + box_padding * 2
    box_width = max_line_width + box_padding * 2

    # Add each text line to the image
    for i, line in enumerate(text_lines):
        text_y = start_y + i * line_height
        cv2.putText(img, line, (start_x + box_padding, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    # Write the frame to the output file
    out.write(img)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img, f"FPS: {fps:.2f}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Face Monitoring", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the logger and release the webcam
logger.stop()
cam.release()
out.release()
cv2.destroyAllWindows()
