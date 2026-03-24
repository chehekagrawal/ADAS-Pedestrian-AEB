import cv2
import dlib
from imutils import face_utils
import yaml
import time
from src.driver_monitoring.compute_ear import compute_ear
from src.driver_monitoring.alertness_state import DriverStateTracker, AlertnessState

class DrowsinessDetector:
    def __init__(self, config_path="configs/part5_thresholds.yaml", predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.tracker = DriverStateTracker(self.config)
        
        # Initialize dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Grab the indexes of the facial landmarks for the left and right eye
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def process_frame(self, frame):
        """
        Process a single BGR frame from webcam.
        Returns the processed frame, current AlertnessState, and Reaction Delay.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        # For speed, we do not upsample
        rects = self.detector(gray, 0)
        
        # If no face is found, we can optionally assume a distracted/unsafe state, 
        # but for this script we just retain the previous state.
        if len(rects) == 0:
             # Face not visible -> treat as unsafe/inattentive for safety
             self.tracker.current_state = AlertnessState.DROWSY
             return frame, self.tracker.current_state, self.tracker.get_reaction_delay()
             
        # Take the largest face (assuming driver is the largest)
        rect = max(rects, key=lambda r: r.area())
        
        # Determine the facial landmarks for the face region
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        
        # Calculate EAR for both eyes
        leftEAR = compute_ear(leftEye)
        rightEAR = compute_ear(rightEye)
        
        # Average the EAR together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Visualize the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Check if EAR is below closed threshold
        is_closed = ear < self.config["ear_thresholds"]["closed_max"]
        
        # Update temporal state
        state = self.tracker.update_state(is_closed)
        delay = self.tracker.get_reaction_delay()
        
        # Overlay information on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"STATE: {state.value}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"DELAY: {delay}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        return frame, state, delay
