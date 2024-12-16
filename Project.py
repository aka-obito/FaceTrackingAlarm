import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading
import sounddevice as sd
import soundfile as sf
import random

# Global variables
is_face_detected = False
stop_detection = False
buzzer_event = threading.Event()

def play_buzzer():
    """
    Plays the buzzer sound in a loop while a face is detected.
    Stops immediately if `buzzer_event` is set.
    """
    global is_face_detected
    data, samplerate = sf.read("buzzer.wav")
    while not buzzer_event.is_set():
        if is_face_detected and not stop_detection:
            sd.play(data, samplerate)
            sd.wait()
        else:
            sd.stop()
    sd.stop()

def pose():
    """
    Detect a face, display live camera feed, and capture a random screenshot of a detected face.
    """
    global is_face_detected, stop_detection

    # Mediapipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)

    screenshot_taken = False
    random_screenshot = None

    while cap.isOpened():
        if stop_detection:
            break

        success, image = cap.read()
        if not success:
            st.warning("No frames available from webcam. Check your camera.")
            break

        # Flip the image and convert to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image for face landmarks
        results = face_mesh.process(image)

        # Prepare for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If a face is detected
        if results.multi_face_landmarks:
            is_face_detected = True

            # Draw the face mesh on the image
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)

            # Randomly capture a screenshot if not already taken
            if not screenshot_taken and random.random() < 0.1:
                random_screenshot = image.copy()
                screenshot_taken = True

        else:
            is_face_detected = False

        # Display the frame in Streamlit
        frame_placeholder.image(image, channels="BGR", use_container_width=True)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Display the random screenshot if captured
    if random_screenshot is not None:
        screenshot_placeholder.image(random_screenshot, channels="BGR", caption="Random Screenshot", use_container_width=True)

# Streamlit interface
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔒 Face Tracker Anti-Theft Alarm Application</h1>", unsafe_allow_html=True)

# Instructions
st.markdown("<h3 style='color: #2196F3;'>📊 Instructions:</h3>", unsafe_allow_html=True)
st.write("1. 🛠️ Ensure your webcam is connected.")
st.write("2. 🎥 The application will detect faces and display the live feed.")
st.write("3. ⏹️ Use the **Stop Detection** button to end the detection process.")
st.markdown("---")

# Status indicator
status_placeholder = st.empty()

def update_status():
    if is_face_detected:
        status_placeholder.markdown("<p style='color: green; font-size: 18px;'>😅 Face Detected</p>", unsafe_allow_html=True)
    else:
        status_placeholder.markdown("<p style='color: red; font-size: 18px;'>🚫 No Face Detected</p>", unsafe_allow_html=True)

# Placeholders for video feed and screenshot
frame_placeholder = st.empty()
screenshot_placeholder = st.empty()

# Buttons for starting and stopping
button_col1, button_col2 = st.columns(2)
with button_col1:
    if st.button("🔊 Start Face Tracking", use_container_width=True):
        stop_detection = False
        is_face_detected = False
        screenshot_placeholder.empty()

        # Clear the stop event and start the buzzer in a separate thread
        buzzer_event.clear()
        buzzer_thread = threading.Thread(target=play_buzzer)
        buzzer_thread.daemon = True
        buzzer_thread.start()

        # Start the pose detection
        pose()

        # Ensure the buzzer stops when detection ends
        buzzer_event.set()
        update_status()

with button_col2:
    if st.button("❌ Stop Detection", use_container_width=True):
        stop_detection = True
        is_face_detected = False
        buzzer_event.set()
        update_status()
