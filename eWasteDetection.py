import streamlit as st
import base64
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

st.title("Webcam Object Detection")

# Load the trained model
model = YOLO('best (1).pt')  # Update with your model path

def process_frame(image_data):
    image_data = image_data.split(',')[1]  # Remove the data URL part
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Convert to OpenCV format
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

    # Convert frame back to image format for display
    _, buffer = cv2.imencode('.jpg', frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{processed_image}'

# Display video processing results
if st.checkbox('Run Webcam'):
    st.markdown("""
        <video id="video" width="640" height="480" autoplay></video>
        <script>
            var video = document.getElementById('video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    video.srcObject = stream;
                })
                .catch(function(err) {
                    console.error('Error accessing webcam:', err);
                });
        </script>
    """, unsafe_allow_html=True)

    # Video processing loop
    st.markdown("""
        <script>
            const videoElement = document.getElementById('video');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const frameRate = 30;

            function processFrame() {
                if (videoElement.readyState >= 2) {
                    canvas.width = videoElement.videoWidth;
                    canvas.height = videoElement.videoHeight;
                    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                    const frameData = canvas.toDataURL('image/jpeg');

                    fetch('/process-frame', {
                        method: 'POST',
                        body: JSON.stringify({ image: frameData }),
                        headers: { 'Content-Type': 'application/json' }
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Update the video stream placeholder with the processed frame
                        st.image(data.processed_image_url);
                    })
                    .catch(error => console.error('Error processing frame:', error));
                }
                setTimeout(processFrame, 1000 / frameRate);
            }
            processFrame();
        </script>
    """, unsafe_allow_html=True)
