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

# Check if an image was captured
if image is None:
    st.error("No image captured. Please check your webcam and try again.")
else:
    # Convert image to NumPy array if it's not already
    if isinstance(image, np.ndarray):
        frame = image
    else:
        frame = np.array(image)

    # Check the shape of the image
    st.write(f"Image shape: {frame.shape}")

    if frame.size == 0:
        st.error("The captured image is empty. Please try again.")
    else:
        # Handle different image formats
        if len(frame.shape) == 2:  # If the image is grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel but 3D array
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) != 3 or frame.shape[2] != 3:
            st.error(f"Unexpected image format: {frame.shape}")
        else:
            # Process the frame using YOLO model
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
