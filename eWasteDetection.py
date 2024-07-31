import streamlit as st

st.title("Webcam Object Detection")

# Create a checkbox for starting/stopping the webcam
run = st.checkbox('Run Webcam')

# Create a placeholder for the video frames
frame_placeholder = st.empty()

# JavaScript to get live video feed from webcam
if run:
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
    
    st.write("The webcam is running. Processing video feed...")

    # Placeholder for video stream
    video_stream_placeholder = st.empty()

    st.markdown("""
        <script>
            const videoElement = document.getElementById('video');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const frameRate = 30; // Adjust the frame rate if necessary

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
                        video_stream_placeholder.image(data.processed_image_url);
                    })
                    .catch(error => console.error('Error processing frame:', error));
                }
                setTimeout(processFrame, 1000 / frameRate);
            }
            processFrame();
        </script>
    """, unsafe_allow_html=True)

    # Backend endpoint for processing video frames
    import flask
    from flask import request, jsonify
    import cv2
    
    from PIL import Image
    from io import BytesIO
    from ultralytics import YOLO

    app = flask.Flask(__name__)

    # Load the trained model
    model = YOLO(r'C:\Users\USER\Documents\imageprocessing2\1435images30epochs\best (1).pt')

    @app.route('/process-frame', methods=['POST'])
    def process_frame():
        data = request.get_json()
        image_data = data['image']
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

        # Convert frame to a format that can be displayed in the browser
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'processed_image_url': f'data:image/jpeg;base64,{processed_image}'})

    if __name__ == '__main__':
        app.run(debug=True, port=5000)
