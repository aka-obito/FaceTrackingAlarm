import cv2
import mediapipe as mp
import numpy as np
import threading
import sounddevice as sd
import soundfile as sf

# Global variables
is_face_detected = False
video_writer = None
video_started = False

def play_buzzer():
    """
    Plays the buzzer sound in a loop while a face is detected.
    """
    global is_face_detected
    data, samplerate = sf.read("buzzer.wav")
    while True:
        if is_face_detected:
            sd.play(data, samplerate)
            sd.wait()  # Wait for the sound to finish
        else:
            sd.stop()  # Stop playing if no face is detected

def pose():
    """
    Detect a face and record a single video when a face is detected.
    """
    global is_face_detected, video_writer, video_started

    # Mediapipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

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

            # Start video recording if it hasn't started
            if not video_started:
                video_name = "detected_face.avi"
                video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
                print(f"Recording started: {video_name}")
                video_started = True

            # Write the frame to the video
            if video_writer:
                video_writer.write(image)

            # Draw the face mesh on the image
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
        else:
            is_face_detected = False

        # Display the frame
        cv2.imshow('Face Tracker with Pose', image)

        # Break on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
        print("Recording stopped.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the buzzer in a separate thread
    buzzer_thread = threading.Thread(target=play_buzzer)
    buzzer_thread.daemon = True
    buzzer_thread.start()

    # Start the pose detection
    pose()
