import cv2
from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image

# Load the trained model
model = YOLO('best (1).pt')

st.title("Webcam Object Detection")

# Create a checkbox for starting/stopping the webcam
run = st.checkbox('Run Webcam')

# Create a placeholder for the video frames
frame_placeholder = st.empty()

# Use session state to maintain the state of the checkbox
if 'run' not in st.session_state:
    st.session_state.run = False

# Update session state based on checkbox
st.session_state.run = run

# Function to read from the webcam and display the video frames
def webcam_stream():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Failed to open webcam.")
        return

    while st.session_state.run:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes on the frame
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cls = int(box[5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the frame to RGB format and display it using Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_image)

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Start the webcam stream in a new thread if the checkbox is checked
if st.session_state.run:
    webcam_stream()
    st.stop()

st.write("Click the checkbox to start the webcam.")
