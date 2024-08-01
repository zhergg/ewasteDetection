import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
from ultralytics import YOLO
import av

# Load the YOLO model
model = YOLO('best (1).pt')  # Replace with your actual model path

st.title("Webcam Object Detection")

# RTC configuration to allow the camera to work on more browsers
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the frame using YOLO model
        results = self.model(img)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cls = int(box[5])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{self.model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,  # Use WebRtcMode.SENDRECV instead of mode.name
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
