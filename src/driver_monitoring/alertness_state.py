from enum import Enum
import time

class AlertnessState(Enum):
    ALERT = "ALERT"
    TIRED = "TIRED"
    DROWSY = "DROWSY"
    MICROSLEEP = "MICROSLEEP"

class DriverStateTracker:
    def __init__(self, thresholds_config):
        self.config = thresholds_config
        self.current_state = AlertnessState.ALERT
        
        self.eyes_closed_start_time = None
        self.consecutive_blinks = 0
        self.last_blink_time = time.time()
        
    def update_state(self, is_eyes_closed):
        current_time = time.time()
        
        if is_eyes_closed:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
                
            duration = current_time - self.eyes_closed_start_time
            
            if duration > self.config["temporal_windows"]["microsleep_min_duration"]:
                self.current_state = AlertnessState.MICROSLEEP
            elif duration > self.config["temporal_windows"]["blink_max_duration"]:
                # If closed more than a normal blink but less than microsleep
                self.current_state = AlertnessState.DROWSY
                
        else:
            if self.eyes_closed_start_time is not None:
                # Eyes just opened
                duration = current_time - self.eyes_closed_start_time
                
                # Check for rapid blinking (signs of tiredness)
                if duration <= self.config["temporal_windows"]["blink_max_duration"]:
                    if current_time - self.last_blink_time < 2.0: # 2 seconds between blinks
                        self.consecutive_blinks += 1
                    else:
                        self.consecutive_blinks = 1
                        
                    self.last_blink_time = current_time
                    
                    if self.consecutive_blinks > 5:
                        self.current_state = AlertnessState.TIRED
                    elif self.current_state not in [AlertnessState.DROWSY, AlertnessState.MICROSLEEP]:
                         self.current_state = AlertnessState.ALERT
                
                self.eyes_closed_start_time = None
            else:
                # Reset tired state if eyes are open and not blinking rapidly
                if current_time - self.last_blink_time > 5.0 and self.current_state == AlertnessState.TIRED:
                    self.current_state = AlertnessState.ALERT
                    self.consecutive_blinks = 0
                
                # If they were in microsleep or drowsy, recovering takes longer or manual reset.
                # Here we slowly recover to ALERT.
                if self.current_state in [AlertnessState.DROWSY, AlertnessState.MICROSLEEP]:
                     if current_time - getattr(self, 'recovery_start', current_time) > 3.0:
                         self.current_state = AlertnessState.ALERT
                     else:
                         self.recovery_start = self.recovery_start if hasattr(self, 'recovery_start') else current_time
                elif self.current_state != AlertnessState.TIRED:
                     self.current_state = AlertnessState.ALERT
                     
        return self.current_state
        
    def get_reaction_delay(self):
        delays = self.config["reaction_mapping"]
        return delays.get(self.current_state.value, 0.7)
