import threading

import av
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

from audio_handling import AudioHandler
from user_defined import UserProcessFrame, get_mediapipe_app

user_process_frame = UserProcessFrame()

alarm_file_path = r"audio/wake_up_og.wav"
audio_handler = AudioHandler(sound_file_path=alarm_file_path)

lock = threading.Lock()  # thread-safe access & prevent race condition.

st.title("Drowsiness Detection!")

col1, col2 = st.columns(2)

with col1:
    EAR_THRESH = st.slider("Eye Aspect Ratio threshold:", 0.0, 0.4, 0.18, 0.01)

with col2:
    WAIT_TIME = st.slider("Seconds to wait before sounding alarm:", 0.0, 5.0, 1.0, 0.25)


# Initialize face mesh solution
face_mesh = get_mediapipe_app()


thresholds = {
    "EAR_THRESH": EAR_THRESH,
    "WAIT_TIME": WAIT_TIME,
}

shared_state = {"play_alarm": False}


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="bgr24")  # Decode and get RGB frame

    frame, play_alarm = user_process_frame.process(frame, face_mesh, thresholds)  # Process frame
    with lock:
        shared_state["play_alarm"] = play_alarm  # Update shared state

    return av.VideoFrame.from_ndarray(frame, format="bgr24")  # Encode and return BGR frame


def audio_frame_callback(frame: av.AudioFrame):
    with lock:
        play_alarm = shared_state["play_alarm"]

    new_frame: av.AudioFrame = audio_handler.process_audio_frame(frame, play_sound=play_alarm)
    return new_frame


# https://github.com/whitphx/streamlit-webrtc/blob/main/streamlit_webrtc/config.py

ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_frame_callback=video_frame_callback,
    audio_frame_callback=audio_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
    media_stream_constraints={"video": {"width": {"ideal": 480}, "height": {"ideal": 480}}, "audio": True},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
)
# -----------------------------------------------------

#
# ctx = webrtc_streamer(
#     key="vpf",
#     video_processor_factory=VideoProcessor,
#     async_processing=True,
#     video_html_attrs=VideoHTMLAttributes(
#         autoPlay=True, controls=False, style={"width": "100%"}, muted=False
#     ),
#     # media_stream_constraints={
#     #     "video": {"width": {"min": 240}, "height": {"min": 240}, "audio": False}
#     # },
# )
