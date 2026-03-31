"""
Unit Tests for Part 2 — Perception & Preprocessing Layer.

Tests all 4 modules:
    1. CLAHE Night Vision Enhancement
    2. Head Pose & Gaze Detection
    3. Lane Detection & In-Path Filtering
    4. Weather & Visibility Detection

Run: python -m pytest tests/test_part2_perception.py -v
"""

import sys
import os
import numpy as np
import cv2
import pytest

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ═══════════════════════════════════════════════════════════════════
# Module 1: CLAHE Night Vision Enhancement
# ═══════════════════════════════════════════════════════════════════

class TestNightEnhance:
    """Tests for src/preprocessing/night_enhance.py"""

    def test_enhance_frame_shape_preserved(self):
        """CLAHE output should have the same shape as input."""
        from src.preprocessing.night_enhance import enhance_frame
        frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        enhanced = enhance_frame(frame)
        assert enhanced.shape == frame.shape
        assert enhanced.dtype == frame.dtype

    def test_enhance_frame_brightens_dark_image(self):
        """CLAHE should increase mean brightness of a dark frame."""
        from src.preprocessing.night_enhance import enhance_frame
        dark_frame = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
        enhanced = enhance_frame(dark_frame)
        dark_mean = np.mean(cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY))
        enhanced_mean = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        assert enhanced_mean > dark_mean, "Enhanced frame should be brighter"

    def test_is_low_light_dark_frame(self):
        """Dark frame (mean < 80) should return True."""
        from src.preprocessing.night_enhance import is_low_light
        dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
        assert is_low_light(dark_frame) is True

    def test_is_low_light_bright_frame(self):
        """Bright frame (mean > 80) should return False."""
        from src.preprocessing.night_enhance import is_low_light
        bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        assert is_low_light(bright_frame) is False

    def test_adaptive_enhance_dark(self):
        """Adaptive enhance should activate on dark frames."""
        from src.preprocessing.night_enhance import adaptive_enhance
        dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        result, was_enhanced = adaptive_enhance(dark_frame)
        assert was_enhanced is True

    def test_adaptive_enhance_bright(self):
        """Adaptive enhance should skip bright frames."""
        from src.preprocessing.night_enhance import adaptive_enhance
        bright_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result, was_enhanced = adaptive_enhance(bright_frame)
        assert was_enhanced is False

    def test_get_brightness(self):
        """Brightness value should be in expected range."""
        from src.preprocessing.night_enhance import get_brightness
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        brightness = get_brightness(frame)
        assert 120 < brightness < 140  # Approximate due to color conversion

    def test_generate_comparison(self, tmp_path):
        """Comparison image should be 2x width and saved to disk."""
        from src.preprocessing.night_enhance import generate_comparison
        frame = np.random.randint(0, 50, (100, 200, 3), dtype=np.uint8)
        save_path = str(tmp_path / "comparison.png")
        comparison = generate_comparison(frame, save_path)
        assert comparison.shape == (100, 400, 3)  # 2x width
        assert os.path.exists(save_path)


# ═══════════════════════════════════════════════════════════════════
# Module 2: Head Pose & Gaze Detection
# ═══════════════════════════════════════════════════════════════════

class TestHeadPose:
    """Tests for src/driver_monitoring/head_pose.py"""

    def _make_landmarks(self):
        """Create synthetic 68-point landmarks (face roughly centered)."""
        landmarks = np.zeros((68, 2), dtype=np.float64)
        # Set the 6 key points used by solvePnP
        landmarks[30] = [320, 240]  # Nose tip
        landmarks[8] = [320, 360]   # Chin
        landmarks[36] = [270, 200]  # Left eye corner
        landmarks[45] = [370, 200]  # Right eye corner
        landmarks[48] = [290, 310]  # Left mouth corner
        landmarks[54] = [350, 310]  # Right mouth corner
        return landmarks

    def test_estimate_head_pose_returns_dict(self):
        """Head pose should return dict with yaw, pitch, roll."""
        from src.driver_monitoring.head_pose import estimate_head_pose
        landmarks = self._make_landmarks()
        result = estimate_head_pose(landmarks, (480, 640))
        assert "yaw" in result
        assert "pitch" in result
        assert "roll" in result
        assert isinstance(result["yaw"], float)
        assert isinstance(result["pitch"], float)
        assert isinstance(result["roll"], float)

    def test_is_distracted_straight(self):
        """Looking straight (yaw=0, pitch=0) should NOT be distracted."""
        from src.driver_monitoring.head_pose import is_distracted
        assert is_distracted(0.0, 0.0) is False

    def test_is_distracted_yaw(self):
        """Large yaw (looking sideways) SHOULD be distracted."""
        from src.driver_monitoring.head_pose import is_distracted
        assert is_distracted(35.0, 0.0) is True
        assert is_distracted(-35.0, 0.0) is True

    def test_is_distracted_pitch(self):
        """Large pitch (looking down) SHOULD be distracted."""
        from src.driver_monitoring.head_pose import is_distracted
        assert is_distracted(0.0, 30.0) is True
        assert is_distracted(0.0, -30.0) is True

    def test_is_distracted_boundary(self):
        """At exactly threshold, should NOT be distracted (strict >)."""
        from src.driver_monitoring.head_pose import is_distracted
        assert is_distracted(30.0, 0.0) is False
        assert is_distracted(0.0, 25.0) is False


# ═══════════════════════════════════════════════════════════════════
# Module 2b: AlertnessState DISTRACTED Integration
# ═══════════════════════════════════════════════════════════════════

class TestAlertnessStateDistracted:
    """Tests for DISTRACTED state in alertness_state.py"""

    def _make_tracker(self):
        """Create a DriverStateTracker with test config."""
        from src.driver_monitoring.alertness_state import DriverStateTracker
        config = {
            "ear_thresholds": {"closed_max": 0.18},
            "temporal_windows": {
                "blink_max_duration": 0.5,
                "microsleep_min_duration": 1.5
            },
            "reaction_mapping": {
                "ALERT": 0.7,
                "TIRED": 1.1,
                "DISTRACTED": 1.3,
                "DROWSY": 1.6,
                "MICROSLEEP": 100.0
            }
        }
        return DriverStateTracker(config)

    def test_distracted_enum_exists(self):
        """DISTRACTED should be a valid AlertnessState."""
        from src.driver_monitoring.alertness_state import AlertnessState
        assert hasattr(AlertnessState, "DISTRACTED")
        assert AlertnessState.DISTRACTED.value == "DISTRACTED"

    def test_update_head_pose_distracted(self):
        """Head distraction should set state to DISTRACTED."""
        from src.driver_monitoring.alertness_state import AlertnessState
        tracker = self._make_tracker()
        state = tracker.update_head_pose(True)
        assert state == AlertnessState.DISTRACTED

    def test_update_head_pose_recovery(self):
        """Returning head to normal should recover from DISTRACTED."""
        from src.driver_monitoring.alertness_state import AlertnessState
        tracker = self._make_tracker()
        tracker.update_head_pose(True)
        state = tracker.update_head_pose(False)
        assert state == AlertnessState.ALERT

    def test_distracted_reaction_delay(self):
        """DISTRACTED state should have 1.3s reaction delay."""
        tracker = self._make_tracker()
        tracker.update_head_pose(True)
        delay = tracker.get_reaction_delay()
        assert delay == 1.3

    def test_drowsy_overrides_distracted(self):
        """DROWSY (more severe) should not be overridden by DISTRACTED."""
        from src.driver_monitoring.alertness_state import AlertnessState
        tracker = self._make_tracker()
        tracker.current_state = AlertnessState.DROWSY
        state = tracker.update_head_pose(True)
        assert state == AlertnessState.DROWSY  # Should remain DROWSY


# ═══════════════════════════════════════════════════════════════════
# Module 3: Lane Detection & In-Path Filtering
# ═══════════════════════════════════════════════════════════════════

class TestLaneDetector:
    """Tests for src/lane_detection/lane_detector.py"""

    def test_create_roi_mask_shape(self):
        """ROI mask should match frame dimensions."""
        from src.lane_detection.lane_detector import _create_roi_mask
        mask = _create_roi_mask((480, 640, 3))
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_create_roi_mask_has_content(self):
        """ROI mask should have some white (255) pixels."""
        from src.lane_detection.lane_detector import _create_roi_mask
        mask = _create_roi_mask((480, 640, 3))
        assert np.sum(mask > 0) > 0

    def test_classify_lines_separation(self):
        """Lines should be separated by slope into left and right."""
        from src.lane_detection.lane_detector import _classify_lines
        # Simulate HoughLinesP output: shape (N, 1, 4)
        lines = np.array([
            [[100, 400, 300, 200]],   # negative slope, left side → left lane
            [[500, 200, 600, 400]],   # positive slope, right side → right lane
        ])
        left, right = _classify_lines(lines, 640)
        assert len(left) > 0, "Should detect left lane"
        assert len(right) > 0, "Should detect right lane"

    def test_classify_lines_filters_horizontal(self):
        """Near-horizontal lines (|slope| < 0.3) should be filtered out."""
        from src.lane_detection.lane_detector import _classify_lines
        lines = np.array([
            [[100, 300, 500, 310]],   # Nearly horizontal, slope ≈ 0.025
        ])
        left, right = _classify_lines(lines, 640)
        assert len(left) == 0
        assert len(right) == 0

    def test_detect_lanes_empty_frame(self):
        """Black frame should return empty list (no lanes)."""
        from src.lane_detection.lane_detector import detect_lanes
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lanes = detect_lanes(black_frame)
        assert isinstance(lanes, list)


class TestEgoCorridor:
    """Tests for src/lane_detection/ego_corridor.py"""

    def test_compute_ego_corridor_valid(self):
        """Valid lanes should produce a valid shapely Polygon."""
        from src.lane_detection.ego_corridor import compute_ego_corridor
        left_lane = [(100, 480), (280, 288)]
        right_lane = [(540, 480), (360, 288)]
        corridor = compute_ego_corridor(left_lane, right_lane, 480)
        assert corridor is not None
        assert corridor.is_valid
        assert corridor.area > 0

    def test_compute_ego_corridor_none_lane(self):
        """Missing lane should return None."""
        from src.lane_detection.ego_corridor import compute_ego_corridor
        corridor = compute_ego_corridor(None, [(540, 480), (360, 288)], 480)
        assert corridor is None

    def test_is_in_path_inside(self):
        """Object inside corridor should return True."""
        from src.lane_detection.ego_corridor import is_in_path, compute_ego_corridor
        left_lane = [(100, 480), (280, 288)]
        right_lane = [(540, 480), (360, 288)]
        corridor = compute_ego_corridor(left_lane, right_lane, 480)
        # Bbox centered in the corridor
        bbox = [280, 350, 360, 450]
        assert is_in_path(bbox, corridor) is True

    def test_is_in_path_outside(self):
        """Object on the sidewalk (outside corridor) should return False."""
        from src.lane_detection.ego_corridor import is_in_path, compute_ego_corridor
        left_lane = [(100, 480), (280, 288)]
        right_lane = [(540, 480), (360, 288)]
        corridor = compute_ego_corridor(left_lane, right_lane, 480)
        # Bbox far to the left (sidewalk)
        bbox = [10, 350, 60, 450]
        assert is_in_path(bbox, corridor) is False

    def test_is_in_path_no_corridor(self):
        """No corridor (None) should default to True (safety-first)."""
        from src.lane_detection.ego_corridor import is_in_path
        bbox = [100, 200, 200, 300]
        assert is_in_path(bbox, None) is True

    def test_filter_detections(self):
        """Filter should partition detections into in-path and off-path."""
        from src.lane_detection.ego_corridor import filter_detections, compute_ego_corridor
        left_lane = [(100, 480), (280, 288)]
        right_lane = [(540, 480), (360, 288)]
        corridor = compute_ego_corridor(left_lane, right_lane, 480)

        detections = [
            {"bbox": [300, 350, 340, 450], "class": "person"},  # In path
            {"bbox": [10, 350, 60, 450], "class": "person"},    # Sidewalk
            {"bbox": [600, 350, 630, 450], "class": "person"},  # Far right
        ]
        in_path, off_path = filter_detections(detections, corridor)
        assert len(in_path) == 1
        assert len(off_path) == 2


# ═══════════════════════════════════════════════════════════════════
# Module 4: Weather & Visibility Detection
# ═══════════════════════════════════════════════════════════════════

class TestWeatherClassifier:
    """Tests for src/weather/weather_classifier.py"""

    def test_heuristic_classify_night(self):
        """Very dark frame should be classified as Night."""
        from src.weather.weather_classifier import WeatherClassifier
        dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        result = WeatherClassifier.heuristic_classify(dark_frame)
        assert result["weather"] == "Night"
        assert 0.0 < result["confidence"] <= 1.0

    def test_heuristic_classify_clear(self):
        """Bright, high-contrast frame should be classified as Clear."""
        from src.weather.weather_classifier import WeatherClassifier
        # Create a bright, varied frame (not fog, not night)
        bright_frame = np.random.randint(150, 255, (100, 100, 3), dtype=np.uint8)
        result = WeatherClassifier.heuristic_classify(bright_frame)
        assert result["weather"] == "Clear"

    def test_classify_with_no_model(self):
        """Classifier without model should use heuristic and include method key."""
        from src.weather.weather_classifier import WeatherClassifier
        classifier = WeatherClassifier(model_path=None)
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = classifier.classify(frame)
        assert "weather" in result
        assert "confidence" in result
        assert result["method"] == "heuristic"

    def test_valid_classes(self):
        """All CLASSES should be defined."""
        from src.weather.weather_classifier import WeatherClassifier
        assert len(WeatherClassifier.CLASSES) == 4
        assert "Clear" in WeatherClassifier.CLASSES
        assert "Rain" in WeatherClassifier.CLASSES
        assert "Fog" in WeatherClassifier.CLASSES
        assert "Night" in WeatherClassifier.CLASSES


class TestWeatherToPhysics:
    """Tests for src/weather/weather_to_physics.py"""

    def test_weather_to_surface_all(self):
        """All weather classes should map to valid tire_model surface names."""
        from src.weather.weather_to_physics import weather_to_surface
        assert weather_to_surface("Clear") == "dry"
        assert weather_to_surface("Rain") == "wet"
        assert weather_to_surface("Fog") == "wet"
        assert weather_to_surface("Night") == "dry"

    def test_weather_to_surface_unknown(self):
        """Unknown weather should default to 'dry' (safe assumption)."""
        from src.weather.weather_to_physics import weather_to_surface
        assert weather_to_surface("Unknown") == "dry"

    def test_weather_to_reaction_penalty(self):
        """Reaction penalties should match specification."""
        from src.weather.weather_to_physics import weather_to_reaction_penalty
        assert weather_to_reaction_penalty("Clear") == 0.0
        assert weather_to_reaction_penalty("Rain") == 0.3
        assert weather_to_reaction_penalty("Fog") == 0.5
        assert weather_to_reaction_penalty("Night") == 0.2

    def test_get_physics_params_returns_all_keys(self):
        """get_physics_params should return dict with all required keys."""
        from src.weather.weather_to_physics import get_physics_params
        for weather in ["Clear", "Rain", "Fog", "Night"]:
            params = get_physics_params(weather)
            assert "surface" in params
            assert "reaction_penalty" in params
            assert "visibility_range" in params
            assert isinstance(params["surface"], str)
            assert isinstance(params["reaction_penalty"], float)
            assert isinstance(params["visibility_range"], float)

    def test_get_physics_params_rain(self):
        """Rain should give wet surface, 0.3s penalty, 60m visibility."""
        from src.weather.weather_to_physics import get_physics_params
        params = get_physics_params("Rain")
        assert params["surface"] == "wet"
        assert params["reaction_penalty"] == 0.3
        assert params["visibility_range"] == 60.0

    def test_get_physics_params_fog(self):
        """Fog should give wet surface, 0.5s penalty, 30m visibility."""
        from src.weather.weather_to_physics import get_physics_params
        params = get_physics_params("Fog")
        assert params["surface"] == "wet"
        assert params["reaction_penalty"] == 0.5
        assert params["visibility_range"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
