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
    while not buzzer_event.is_set():  # Check if the event is set to stop the thread
        if is_face_detected and not stop_detection:
            sd.play(data, samplerate)
            sd.wait()  # Wait for the sound to finish
        else:
            sd.stop()  # Stop playing if no face is detected or detection is stopped
    sd.stop()  # Ensure the buzzer stops completely when exiting

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
            if not screenshot_taken and random.random() < 0.1:  # 10% chance per frame
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
st.title("Face Tracker Anti-theift Alarm Application")

# Instructions
st.markdown("### Instructions:")
st.write("1. Ensure your webcam is connected.\n2. The application will detect faces and display the live feed. \n3. Use the **Stop Detection** button to end the detection process.")

# Placeholders for video feed and screenshot
frame_placeholder = st.empty()
screenshot_placeholder = st.empty()

# Button to start and stop the face tracking
if st.button("Start Face Tracking"):
    stop_detection = False
    is_face_detected = False  # Reset face detection
    screenshot_placeholder.empty()  # Clear any previous screenshot

    # Clear the stop event and start the buzzer in a separate thread
    buzzer_event.clear()
    buzzer_thread = threading.Thread(target=play_buzzer)
    buzzer_thread.daemon = True
    buzzer_thread.start()

    # Start the pose detection
    pose()

    # Ensure the buzzer stops when detection ends
    buzzer_event.set()

if st.button("Stop Detection"):
    stop_detection = True
    is_face_detected = False  # Ensure face detection stops
    buzzer_event.set()  # Signal the buzzer thread to stop
