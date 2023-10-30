import os
import pprint

# TODO disabled unresolved references
# TODO disabled duplicate

import cv2
from gaze_tracking import GazeTracking
import imutils
import mediapipe as mp
from gaze_tracking import gaze as gz

import time

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
text = "Not Found"
toggle_log = False

if __name__ == "__main__":
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            # We get a new frame from the webcam
            success, frame = webcam.read()
            if not success:  # no frame input
                print(text)

            frame = imutils.resize(frame, width=1600)
            frame.flags.writeable = False

            if gaze.debug_mode:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
                results = face_mesh.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV
                if results.multi_face_landmarks:
                    gz.gaze(frame, results.multi_face_landmarks[0])  # gaze estimation

            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)

            frame = gaze.annotated_frame()

            # Display the log box
            if toggle_log:
                queue_text = "Log: "
                y0 = 90  # Starting y-coordinate for the first line
                dy = 30  # Vertical spacing between lines
                cv2.putText(frame, queue_text, (120, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (147, 31, 58), 2)

                if not gaze.anomaly_queue_log2.empty():
                    recent_anomaly = pprint.pformat(gaze.anomaly_queue_log2.get())
                    print(recent_anomaly)
                    for i, line in enumerate(recent_anomaly.split('\n')):
                        y = y0 + i * dy
                        cv2.putText(frame, line, (120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            cv2.imshow("PyGaze", frame)

            key = cv2.waitKey(1)
            match key:
                case 120:
                    gaze.toggle_debug()
                case 122:
                    if not toggle_log:
                        toggle_log = True
                    else:
                        toggle_log = False
                case 27:
                    break

    webcam.release()
    cv2.destroyAllWindows()

    if not os.path.exists('logs'):
        os.makedirs('logs')

    with open(f'logs/log.txt', 'w') as file:
        # Iterate over the items in the queue and write them to the file
        while not gaze.anomaly_queue_log.empty():
            item = pprint.pformat(gaze.anomaly_queue_log2.get())
            log_entry = f"{time.ctime(time.time())}: \n{item}\n\n"
            file.write(log_entry)
