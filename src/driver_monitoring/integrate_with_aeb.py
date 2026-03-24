import cv2
import argparse
import os
import sys

# Ensure src in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.driver_monitoring.drowsiness_detector import DrowsinessDetector
from src.driver_monitoring.alertness_state import AlertnessState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0, help="Webcam ID or video file path")
    args = parser.parse_args()

    # We need the dlib weight file
    if not os.path.exists("models/shape_predictor_68_face_landmarks.dat"):
        print("ERROR: Could not find models/shape_predictor_68_face_landmarks.dat")
        print("Please download it and place it in the models directory.")
        print("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return

    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    print("Starting driver monitoring... Press 'q' to quit.")

    # Simulating connection to AEB: we will print out what the AEB threshold would be
    base_aeb_threshold = 1.5 # standard TTC threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to detect drowsiness
        processed_frame, state, delay = detector.process_frame(frame)
        
        # The adaptive threshold logic
        adaptive_threshold = base_aeb_threshold + delay
        
        # Display the adaptive threshold
        color = (0, 255, 0)
        if state == AlertnessState.DROWSY:
            color = (0, 165, 255) # Orange
        elif state == AlertnessState.MICROSLEEP:
            color = (0, 0, 255) # Red
            
        cv2.putText(processed_frame, f"AEB TRIGGER THRESHOLD: {adaptive_threshold:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Part 5: Driver Monitoring", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
