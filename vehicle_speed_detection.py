import cv2
import numpy as np
import time
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define video file path
video_path = "traffic.mp4"

# Print current directory for debugging
print("Current directory:", os.getcwd())

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video '{video_path}'.")
    exit()

# Get the original frame rate of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define a function to calculate speed
def calculate_speed(p1, p2, time_diff, pixel_to_meter_ratio):
    # Calculate distance in pixels
    distance_pixels = np.linalg.norm(np.array(p1) - np.array(p2))
    
    # Convert distance from pixels to meters using pixel_to_meter_ratio
    distance_meters = distance_pixels * pixel_to_meter_ratio
    
    # Calculate speed in meters per second
    speed_mps = distance_meters / time_diff
    
    # Convert speed to kilometers per hour (km/h)
    speed_kmph = speed_mps * 3.6
    
    return speed_kmph

# Define threshold speed for red bounding box (100 km/h)
threshold_speed_kmph = 100

# Example pixel_to_meter_ratio (adjust according to your setup)
pixel_to_meter_ratio = 0.1  # Adjust based on your specific video and object size

# Variables to store vehicle positions and timestamps
positions = {}
timestamps = {}

# Define the region of interest (ROI) for nearby vehicles
# Adjust these coordinates according to your video
roi_top_left = (100, 100)
roi_bottom_right = (500, 500)

# Main loop to process frames
frame_count = 0
process_every_n_frames = 5  # Process every 5th frame to speed up

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        # Skip frames to reduce processing load
        continue

    height, width, channels = frame.shape

    # Reduce the resolution of the frame for faster processing
    resized_frame = cv2.resize(frame, (320, 320))

    # Detecting objects
    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialization
    class_ids = []
    confidences = []
    boxes = []

    # For each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Filter objects within the ROI
                if (roi_top_left[0] <= center_x <= roi_bottom_right[0]) and (roi_top_left[1] <= center_y <= roi_bottom_right[1]):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    current_positions = {}
    current_time = time.time()

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 150, 0)  # Green color for default bounding box color

            if label == "car" or label == "truck" or label == "bus":
                center = (x + w // 2, y + h // 2)
                current_positions[label] = center

                if label in positions:
                    p1 = positions[label]
                    p2 = center
                    time_diff = current_time - timestamps[label]
                    speed = calculate_speed(p1, p2, time_diff, pixel_to_meter_ratio)
                    
                    if speed > threshold_speed_kmph:
                        color = (0, 0, 150)  # Red color for high-speed vehicles
                        
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"{speed:.2f} km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update positions and timestamps
    positions = current_positions
    timestamps = {label: current_time for label in current_positions.keys()}

    # Draw the ROI on the frame
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)  # Blue rectangle for ROI

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#copyright @arunbabu