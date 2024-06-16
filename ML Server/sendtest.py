import os
import cv2
from flask import Flask, Response, jsonify, send_from_directory
from threading import Thread
from collections import deque
import numpy as np
from datetime import datetime
import FCMmanager as fcm

import tensorflow

app = Flask(__name__)

import time

# Variable to keep track of the last time the message was sent
last_notification_time = 0

import time
from collections import deque
import cv2
import numpy as np
import keras
import FCMmanager as fcm  # Assuming this is a module for sending FCM notifications


# Load the pre-trained model
model = keras.models.load_model("modelnew.h5")

# Define the maximum number of frames to keep in the deque
MAX_FRAMES = 128

def detect_fights(video_path):
    vs = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_frames = []
    anomaly_detected = False
    saving_video = False

    while True:
        grabbed, frame = vs.read()

        if not grabbed:
            break

        if not saving_video:
            frame_copy = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128)).astype("float32")
            frame = frame.reshape(128, 128, 3) / 255

            preds = model.predict(np.expand_dims(frame, axis=0))[0]

            if np.mean(preds) > 0.25:
                if not anomaly_detected:
                    anomaly_detected = True
                    start_frame = vs.get(cv2.CAP_PROP_POS_FRAMES) - 1
                    output_frames = []

                output_frames.append(frame_copy)

            elif anomaly_detected:
                anomaly_detected = False
                end_frame = vs.get(cv2.CAP_PROP_POS_FRAMES) - 1
                save_video_clip(output_frames, start_frame, end_frame, fourcc)
                saving_video = True

        else:
            output_frames.append(frame)
            if len(output_frames) >= MAX_FRAMES:
                end_frame = vs.get(cv2.CAP_PROP_POS_FRAMES) - 1
                save_video_clip(output_frames, start_frame, end_frame, fourcc)
                saving_video = False

    vs.release()
    print("Processing finished.")

def save_video_clip(frames, start_frame, end_frame, fourcc):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{timestamp}.avi"
    output_path = os.path.join(os.getcwd(), "output", output_filename)
    height, width, _ = frames[0].shape
    output_video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for frame in frames:
        output_video.write(frame)
    output_video.release()





@app.route('/')
def home():
    return "Hi world"

@app.route('/fetchvideos', methods=['GET'])
def fetchvideos():
    folder = 'output'
    video_files = [f for f in os.listdir(folder) if f.endswith('.avi') or f.endswith('.mp4')]
    video_urls = [{'url': f'/videos/{filename}'} for filename in video_files]
    return jsonify(video_urls)

@app.route('/videos/<path:filename>', methods=['GET'])
def serve_video(filename):
    folder = 'output'
    return send_from_directory(folder, filename)

@app.after_request
def after_request(response):
    video_path = "https://fe4e-2402-3a80-4479-752-e9da-3223-c2fd-b374.ngrok-free.app/stream"
    thread = Thread(target=detect_fights, args=(video_path,))
    thread.start()
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
