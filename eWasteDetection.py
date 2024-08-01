import streamlit as st
from camera_input_live import camera_input_live
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best (1).pt')  # Replace with your actual model path

# Streamlit title
st.title("Webcam Object Detection")

# Capture image from the webcam using the camera_input_live package
image = camera_input_live()

if image:
    # Convert the image to a numpy array
    image = np.array(image)

    # Convert the image to BGR format (as OpenCV uses BGR)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cls = int(box[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the processed frame back to RGB (for Streamlit)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the processed image in the Streamlit app
    st.image(frame, caption="Processed Image")
