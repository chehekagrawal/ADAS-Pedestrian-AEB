import os
import joblib
import numpy as np


class AEBController:
    """
    Physics-aware and machine-learning-augmented AEB Controller.
    
    Supports:
        1. Physics rule-based evaluation: TTC < (TTC_thresh + reaction_time + thermo_penalty)
        2. Machine learning prediction: Random Forest model predicting intervention probability
        3. Hybrid arbitration: Triggers if either rule-based safety envelope or ML model indicates critical risk
    """

    def __init__(self, ttc_threshold=1.5, model_path="models/adaptive_aeb_model.pkl"):
        self.ttc_threshold = ttc_threshold
        self.triggered = False
        self.ml_model = None
        self.feature_columns = ["ttc", "distance", "v_rel", "weather_friction", "driver_ear", "rotor_temp"]

        if model_path and os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                if isinstance(bundle, dict):
                    self.ml_model = bundle.get("model", bundle)
                    self.feature_columns = bundle.get("feature_columns", self.feature_columns)
                else:
                    self.ml_model = bundle
            except Exception:
                self.ml_model = None

    def evaluate(self, ttc, reaction_time=0.0, thermo_penalty=0.0):
        """
        Rule-based adaptive braking logic:
        A higher reaction time or a thermodynamic brake fade penalty
        forces the system to trigger EARLIER (at a higher TTC).
        """
        if ttc < (self.ttc_threshold + reaction_time + thermo_penalty):
            self.triggered = True
        return self.triggered

    def evaluate_ml(
        self,
        ttc: float,
        distance: float,
        v_rel: float,
        weather_friction: float = 0.85,
        driver_ear: float = 0.30,
        rotor_temp: float = 20.0,
    ):
        """
        Evaluate AEB intervention using the trained Random Forest classifier.
        Returns: (triggered: bool, risk_probability: float)
        """
        if self.ml_model is None:
            # Fallback to rule-based evaluation if model not loaded
            rule_trig = self.evaluate(ttc)
            return rule_trig, 1.0 if rule_trig else 0.0

        features = np.array([[ttc, distance, v_rel, weather_friction, driver_ear, rotor_temp]], dtype=np.float32)
        pred = int(self.ml_model.predict(features)[0])
        prob = float(self.ml_model.predict_proba(features)[0, 1]) if hasattr(self.ml_model, "predict_proba") else float(pred)

        if pred == 1:
            self.triggered = True
        return bool(pred == 1), prob

    def evaluate_hybrid(
        self,
        ttc: float,
        distance: float,
        v_rel: float,
        weather_friction: float = 0.85,
        driver_ear: float = 0.30,
        rotor_temp: float = 20.0,
        reaction_time: float = 0.0,
        thermo_penalty: float = 0.0,
        ml_prob_threshold: float = 0.60,
    ):
        """
        Hybrid arbitration: Triggers if either physical safety envelope is breached
        or ML model assesses high probability of collision.
        """
        rule_trig = self.evaluate(ttc, reaction_time=reaction_time, thermo_penalty=thermo_penalty)
        ml_trig, ml_prob = self.evaluate_ml(
            ttc=ttc,
            distance=distance,
            v_rel=v_rel,
            weather_friction=weather_friction,
            driver_ear=driver_ear,
            rotor_temp=rotor_temp,
        )

        final_trigger = rule_trig or (ml_prob >= ml_prob_threshold)
        if final_trigger:
            self.triggered = True
        return final_trigger, {"rule_triggered": rule_trig, "ml_triggered": ml_trig, "ml_probability": ml_prob}

