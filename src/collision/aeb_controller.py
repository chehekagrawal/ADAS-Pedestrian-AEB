class AEBController:
    def __init__(self, ttc_threshold=1.5):
        self.ttc_threshold = ttc_threshold
        self.triggered = False

    def evaluate(self, ttc, reaction_time=0.0, thermo_penalty=0.0):
        # Adaptive braking logic:
        # A higher reaction time or a thermodynamic brake fade penalty
        # forces the system to trigger EARLIER (at a higher TTC)
        if ttc < (self.ttc_threshold + reaction_time + thermo_penalty):
            self.triggered = True
        return self.triggered

