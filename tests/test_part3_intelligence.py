"""
Unit Tests for Part 3 — Intelligence, Explainability & Pipeline Integration.

Tests all modules:
    1. Traffic Sign Recognition & AEB Threshold Adjustment
    2. Black Box Incident Recorder
    3. Pipeline module imports & configuration

Run: python -m pytest tests/test_part3_intelligence.py -v
"""

import sys
import os
import numpy as np
import cv2
import json
import pytest
from collections import deque

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ═══════════════════════════════════════════════════════════════════
# Module 1: Traffic Sign Recognition
# ═══════════════════════════════════════════════════════════════════

class TestTrafficSignDetector:
    """Tests for src/traffic_signs/sign_detector.py"""

    def test_adjust_aeb_threshold_school_zone(self):
        """School zone (30 km/h) should tighten threshold by 50%."""
        from src.traffic_signs.sign_detector import adjust_aeb_threshold
        result = adjust_aeb_threshold(1.5, 30)
        assert result == pytest.approx(2.25)  # 1.5 * 1.5

    def test_adjust_aeb_threshold_urban(self):
        """Urban (50 km/h) should keep normal threshold."""
        from src.traffic_signs.sign_detector import adjust_aeb_threshold
        result = adjust_aeb_threshold(1.5, 50)
        assert result == pytest.approx(1.5)  # 1.5 * 1.0

    def test_adjust_aeb_threshold_highway(self):
        """Highway (80 km/h) should relax threshold by 10%."""
        from src.traffic_signs.sign_detector import adjust_aeb_threshold
        result = adjust_aeb_threshold(1.5, 80)
        assert result == pytest.approx(1.35)  # 1.5 * 0.9

    def test_adjust_aeb_threshold_very_low_speed(self):
        """Very low speed zones should also trigger 1.5x."""
        from src.traffic_signs.sign_detector import adjust_aeb_threshold
        result = adjust_aeb_threshold(1.5, 20)
        assert result == pytest.approx(2.25)

    def test_speed_limits_dict(self):
        """SPEED_LIMITS should contain standard speed values."""
        from src.traffic_signs.sign_detector import TrafficSignDetector
        limits = TrafficSignDetector.SPEED_LIMITS
        assert limits["speed_30"] == 30
        assert limits["speed_50"] == 50
        assert limits["speed_80"] == 80

    def test_detector_init_no_model(self):
        """Detector should initialize without a model (heuristic mode)."""
        from src.traffic_signs.sign_detector import TrafficSignDetector
        detector = TrafficSignDetector(model_path=None)
        assert detector.model is None

    def test_get_active_speed_limit_default(self):
        """Without any sign, should return default 50 km/h urban."""
        from src.traffic_signs.sign_detector import TrafficSignDetector
        detector = TrafficSignDetector(model_path=None)
        # Black frame — no signs
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.get_active_speed_limit(frame)
        assert result["speed_limit_kmh"] == 50
        assert result["zone_type"] == "urban"

    def test_heuristic_detect_returns_list(self):
        """Heuristic detection should return a list."""
        from src.traffic_signs.sign_detector import TrafficSignDetector
        detector = TrafficSignDetector(model_path=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        signs = detector.detect_signs(frame)
        assert isinstance(signs, list)


# ═══════════════════════════════════════════════════════════════════
# Module 3: Black Box Incident Recorder
# ═══════════════════════════════════════════════════════════════════

class TestBlackBoxRecorder:
    """Tests for src/recorder/black_box.py"""

    def test_init_buffer_size(self):
        """Buffer should have correct max size."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=10, fps=30)
        assert recorder.buffer_size == 300  # 10 * 30

    def test_record_frame(self):
        """Recording a frame should increase buffer count."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=5, fps=10)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        recorder.record_frame(frame, {"ttc": 3.0, "aeb_triggered": False})
        assert recorder.buffer_frame_count == 1

    def test_circular_buffer_overflow(self):
        """Buffer should not exceed max size (circular behavior)."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=1, fps=5)  # max 5 frames
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        for i in range(10):
            recorder.record_frame(frame, {"frame": i})

        assert recorder.buffer_frame_count == 5  # Only last 5 kept

    def test_buffer_duration(self):
        """Buffer duration should be computed correctly."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=5, fps=10)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        for i in range(30):
            recorder.record_frame(frame, {"frame": i})

        assert recorder.buffer_duration == pytest.approx(3.0)  # 30 / 10

    def test_save_incident(self, tmp_path):
        """save_incident should create a directory with files."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=1, fps=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        for i in range(5):
            recorder.record_frame(frame, {
                "frame": i,
                "ttc": 3.0 - i * 0.5,
                "aeb_triggered": i == 4,
            })

        incident_dir = recorder.save_incident(str(tmp_path / "incidents"))
        assert incident_dir is not None
        assert os.path.exists(incident_dir)
        assert os.path.exists(os.path.join(incident_dir, "video.mp4"))
        assert os.path.exists(os.path.join(incident_dir, "sensor_log.json"))
        assert os.path.exists(os.path.join(incident_dir, "summary.json"))

    def test_save_incident_json_valid(self, tmp_path):
        """Saved sensor log should be valid JSON."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=1, fps=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        recorder.record_frame(frame, {
            "ttc": 2.5,
            "aeb_triggered": False,
            "numpy_val": np.float64(3.14),
            "numpy_int": np.int32(42),
        })

        incident_dir = recorder.save_incident(str(tmp_path / "incidents"))
        log_path = os.path.join(incident_dir, "sensor_log.json")

        with open(log_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_clear_buffer(self):
        """clear() should empty the buffer."""
        from src.recorder.black_box import BlackBoxRecorder
        recorder = BlackBoxRecorder(buffer_seconds=1, fps=5)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        recorder.record_frame(frame, {"test": 1})
        recorder.clear()
        assert recorder.buffer_frame_count == 0

    def test_make_serializable(self):
        """Numpy types should be converted to native Python."""
        from src.recorder.black_box import BlackBoxRecorder
        data = {
            "np_float": np.float64(3.14),
            "np_int": np.int32(42),
            "np_array": np.array([1, 2, 3]),
            "normal": "string",
            "inf": float("inf"),
        }
        result = BlackBoxRecorder._make_serializable(data)
        assert isinstance(result["np_float"], float)
        assert isinstance(result["np_int"], int)
        assert isinstance(result["np_array"], list)
        assert result["inf"] == "inf"


# ═══════════════════════════════════════════════════════════════════
# Module 2: Grad-CAM (import tests only — needs YOLO model)
# ═══════════════════════════════════════════════════════════════════

class TestGradCAM:
    """Tests for src/explainability/gradcam.py (import and init only)."""

    def test_gradcam_import(self):
        """GradCAM class should be importable."""
        from src.explainability.gradcam import GradCAM
        assert GradCAM is not None

    def test_gradcam_no_model(self):
        """GradCAM without model should not crash."""
        from src.explainability.gradcam import GradCAM
        cam = GradCAM(model_path="nonexistent_model.pt")
        assert cam.model is None

    def test_overlay_heatmap_passthrough(self):
        """overlay_heatmap with None heatmap should return original frame."""
        from src.explainability.gradcam import GradCAM
        cam = GradCAM(model_path="nonexistent.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cam.overlay_heatmap(frame, None)
        assert np.array_equal(result, frame)


# ═══════════════════════════════════════════════════════════════════
# Module 4: Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Tests for run_pipeline.py imports and basic structure."""

    def test_pipeline_import(self):
        """Pipeline module should be importable."""
        import run_pipeline
        assert hasattr(run_pipeline, 'ADASPipeline')
        assert hasattr(run_pipeline, 'run_pipeline')

    def test_weather_to_physics_in_pipeline(self):
        """Weather → physics bridge should work in pipeline context."""
        from src.weather.weather_to_physics import get_physics_params
        from src.traffic_signs.sign_detector import adjust_aeb_threshold

        # Simulate: Rain detected, school zone
        params = get_physics_params("Rain")
        threshold = adjust_aeb_threshold(1.5, 30)

        assert params["surface"] == "wet"
        assert params["reaction_penalty"] == 0.3
        assert threshold == pytest.approx(2.25)

        # Total effective threshold
        total = threshold + 0.7 + params["reaction_penalty"]  # base + driver + weather
        assert total > 3.0  # Should be more aggressive in rain + school zone

    def test_full_chain_aeb_decision(self):
        """Test complete AEB decision chain: weather + zone + driver state."""
        from src.collision.aeb_controller import AEBController
        from src.weather.weather_to_physics import get_physics_params
        from src.traffic_signs.sign_detector import adjust_aeb_threshold

        # Scenario: Fog, school zone, driver is alert
        physics = get_physics_params("Fog")
        adjusted_threshold = adjust_aeb_threshold(1.5, 30)
        driver_delay = 0.7  # ALERT
        weather_penalty = physics["reaction_penalty"]
        total_reaction = driver_delay + weather_penalty

        aeb = AEBController(ttc_threshold=adjusted_threshold)

        # At TTC = 3.0s, should it trigger?
        effective_threshold = adjusted_threshold + total_reaction
        # 2.25 + 0.7 + 0.5 = 3.45 → TTC 3.0 < 3.45 → SHOULD trigger
        triggered = aeb.evaluate(3.0, reaction_time=total_reaction)
        assert triggered is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
