import streamlit as st


def add_1():
    st.sidebar.header("ADD 1")

    # col1, col2, col3 = st.columns([3, 3, 4], gap="medium")

    html_string = '<a href="https://opencv.org/courses" target="_blank"><img src="https://learnopencv.com/wp-content/uploads/2022/03/opencv-course1.png" alt="Opencv Courses"  width="500" height="282"></a>'

    # with col1:
    #     st.markdown(html_string, unsafe_allow_html=True)

    # with col2:
    #     st.markdown(html_string, unsafe_allow_html=True)
    # with col3:

    st.sidebar.markdown(html_string, unsafe_allow_html=True)


class DrowsinessDetectionVideoFrameHandler:
    def __init__(self):
        '''
        Intialize the necessary constants, mediapipe app 
        and tracker variables
        '''
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Used for coloring landmark points.
        # It's value depends on the condition whether the EAR value
        # has dropped below threshold limit or not.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline 
        self.facemesh_model = get_mediapipe_app()


        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": self.GREEN,
            "play_alarm": False,
        }

        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.array, thresholds: dict):
        '''
        This function is used to implement our Drowsy detextion algorithm
        
        Args:
            frame: (np.arrray) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH. 

        Returns:
            The processed frame and a boolean flag to 
            indicate if the alarm should be played or not.
        '''
        
        # To improve performance, 
        # mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))


        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])

            if EAR < thresholds["EAR_THRESH"]:

                # Increase DROWSY_TIME to track the time period with EAR less than threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a selfie-view display.

        return frame, self.state_tracker["play_alarm"]
