import av
import cv2
import time
import threading
import numpy as np
import mediapipe as mp
import streamlit as st
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

# =====================================================
# ======================CONSTANTS======================
# =====================================================

# Left and right eye chosen landmarks.
left_eye_idxs = [362, 385, 387, 263, 373, 380]
right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_idxs = left_eye_idxs + right_eye_idxs  # This list is only used for plotting the landmarks.

# Used for coloring landmark points.
# It's value depends on the condition whether the EAR value
# has dropped below threshold limit or not.

RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)  # BGR

alarm_file_path = "wake_up_og.wav"
# -----------------------------------------------------

# =====================================================
# =================NECESSARY FUNCTIONS=================
# =====================================================


# @st.cache(allow_output_mutation=True)
def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6

        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye apect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=1.0, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


# -----------------------------------------------------

# =====================================================
# =================STREAMLIT COMPONENTS================
# =====================================================

st.title("Drowsiness Detection!")

col1, col2 = st.columns(2)

with col1:
    EAR_THRESH = st.slider("Eye Aspect Ratio threshold:", 0.0, 0.4, 0.18, 0.01)

with col2:
    WAIT_TIME = st.slider("Seconds to wait before sounding alarm:", 0.0, 5.0, 1.0, 0.25)


# -----------------------------------------------------

# =====================================================
# ==============Image and Audio Processing=============
# =====================================================

# Initialize face mesh solution
face_mesh = get_mediapipe_app()

lock = threading.Lock()  # For updating states

# For tracking counters and sharing states in and out of callbacks.
state_tracker = {
    "play_alarm": False,
    "start_time": time.perf_counter(),
    "DROWSY_TIME": 0.0,
    "COLOR": GREEN,
}

# For audio playing
play_state_tracker = {"curr_segment": -1}

wav_file = AudioSegment.from_file(file=alarm_file_path, format="wav")
wav_file = wav_file.set_channels(2)
wav_file = wav_file.set_frame_rate(48000)
wav_file = wav_file.set_sample_width(2)

ms_per_audio_segment = 20  # in milliseconds

audio_segments = [
    wav_file[i : i + ms_per_audio_segment] for i in range(0, len(wav_file) - len(wav_file) % ms_per_audio_segment, ms_per_audio_segment)
]
total_segments = len(audio_segments) - 1  # -1 because we start from 0.


def video_frame_callback(frame: av.VideoFrame):

    RESET_STATE = False
    COLOR = state_tracker["COLOR"]

    frame = frame.to_ndarray(format="bgr24")
    cam_h, cam_w, _ = frame.shape

    EAR_txt_pos = (int(cam_w // 2 * 1.3), 30)
    ALM_txt_pos = (10, cam_h - 50)
    DROWSY_TIME_txt_pos = (10, cam_h - 100)

    frame.flags.writeable = False

    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, cam_w, cam_h)
        right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, cam_w, cam_h)
        EAR = (left_ear + right_ear) / 2.0

        for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
            if lm_coordinates:
                for coord in lm_coordinates:
                    cv2.circle(frame, coord, 2, COLOR, -1)

        frame = cv2.flip(frame, 1)

        if EAR < EAR_THRESH:

            # Increase DROWSY_TIME to track the time period with EAR less than threshold
            # and reset the start_time for the next iteration.
            end_time = time.perf_counter()
            COLOR = RED

            # with lock:
            state_tracker["DROWSY_TIME"] += end_time - state_tracker["start_time"]
            state_tracker["start_time"] = end_time
            state_tracker["COLOR"] = COLOR

            DROWSY_TIME = state_tracker["DROWSY_TIME"]

            if DROWSY_TIME >= WAIT_TIME:
                plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, COLOR)

                # with lock:
                state_tracker["play_alarm"] = True

        else:
            _timer = time.perf_counter()
            DROWSY_TIME = 0.0
            COLOR = GREEN
            RESET_STATE = True

        plot_text(frame, f"EAR: {round(EAR, 2)}", EAR_txt_pos, COLOR)
        plot_text(frame, f"DROWSY: {round(DROWSY_TIME, 3)} Secs", DROWSY_TIME_txt_pos, COLOR)

    else:
        _timer = time.perf_counter()
        COLOR = GREEN
        RESET_STATE = True

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a selfie-view display.

    if RESET_STATE:
        # with lock:
        state_tracker["COLOR"] = COLOR
        state_tracker["play_alarm"] = False
        state_tracker["start_time"] = _timer
        state_tracker["DROWSY_TIME"] = 0.0

    return av.VideoFrame.from_ndarray(frame, format="bgr24")  # av.VideoFrame


def process_audio(frame: av.AudioFrame):

    raw_samples = frame.to_ndarray()

    _play_alarm = state_tracker["play_alarm"]
    _curr_segment = play_state_tracker["curr_segment"]

    if _play_alarm:
        if _curr_segment < total_segments:
            _curr_segment += 1
        else:
            _curr_segment = 0

        sound = audio_segments[_curr_segment]

    else:
        if -1 < _curr_segment < total_segments:
            _curr_segment += 1
            sound = audio_segments[_curr_segment]

        else:
            _curr_segment = -1
            sound = AudioSegment(
                data=raw_samples.tobytes(),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )
            sound = sound.apply_gain(-100)

    # with lock:
    play_state_tracker["curr_segment"] = _curr_segment

    channel_sounds = sound.split_to_mono()
    channel_samples = [s.get_array_of_samples() for s in channel_sounds]

    new_samples = np.array(channel_samples).T

    new_samples = new_samples.reshape(raw_samples.shape)
    new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
    new_frame.sample_rate = frame.sample_rate

    return new_frame  # av.AudioFrame


# -----------------------------------------------------

# =====================================================
# ====================WEBRTC STREAM====================
# =====================================================

# https://github.com/whitphx/streamlit-webrtc/blob/main/streamlit_webrtc/config.py

ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_frame_callback=video_frame_callback,
    audio_frame_callback=process_audio,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
    media_stream_constraints={"video": {"width": {"min": 480}, "height": {"min": 480}}, "audio": True},
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
