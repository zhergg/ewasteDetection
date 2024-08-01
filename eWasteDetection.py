import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('path/to/your/model/best.pt')  # Replace with your actual model path

# Streamlit title
st.title("Webcam Object Detection")

# Checkbox to start/stop the webcam
run = st.checkbox('Run Webcam')

# Placeholder for the video feed
frame_placeholder = st.empty()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot open webcam")

while run:
    ret, frame = cap.read()

    if not ret:
        st.error("Failed to grab frame")
        break

    # Convert the frame from BGR (OpenCV default) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check the shape of the image
    st.write(f"Image shape: {frame_rgb.shape}")

    if frame_rgb.size == 0:
        st.error("The captured image is empty. Please try again.")
    else:
        # Process the frame using the YOLO model
        results = model(frame_rgb)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cls = int(box[5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the processed frame back to RGB for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed image in the Streamlit app
        frame_placeholder.image(frame_rgb, channels="RGB")

# Release the video capture object when done
cap.release()
