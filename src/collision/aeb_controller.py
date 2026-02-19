class AEBController:
    def __init__(self, ttc_threshold=1.5):
        self.ttc_threshold = ttc_threshold
        self.triggered = False

    def evaluate(self, ttc):
        if ttc < self.ttc_threshold:
            self.triggered = True
        return self.triggered
