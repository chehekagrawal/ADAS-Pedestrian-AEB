# """
# ADAS Master Pipeline — runs the complete system end-to-end.

# Integrates ALL modules from Parts 1–5 into a single per-frame processing loop:
#     1. Preprocessing (CLAHE night vision)
#     2. Detection (YOLO pedestrian/vehicle detection)
#     3. Tracking (SORT multi-object tracking)
#     4. Depth estimation (MiDaS monocular depth)
#     5. Lane detection & ego corridor filtering
#     6. Weather classification → physics parameters
#     7. Traffic sign recognition → dynamic AEB thresholds
#     8. Driver monitoring (drowsiness + head pose → reaction time)
#     9. Collision prediction (TTC, vehicle dynamics, AEB decision)
#     10. Impact analysis (HIC if collision unavoidable)
#     11. Black box recording & incident preservation
#     12. Frame annotation & output

# Usage:
#     python run_pipeline.py --video data/sample/test_video.mp4 --speed 50
#     python run_pipeline.py --video input.mp4 --speed 40 --surface wet
#     python run_pipeline.py --webcam --speed 30

# Output:
#     results/pipeline/
#     ├── output_video.mp4        (annotated video with all overlays)
#     ├── pipeline_log.json       (per-frame metrics)
#     └── pipeline_summary.json   (aggregate statistics)
# """

# import argparse
# import cv2
# import json
# import os
# import sys
# import time
# import numpy as np
# from datetime import datetime

# # ─── Add project root to path ─────────────────────────────────────
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# # ─── Module imports (with graceful fallbacks) ──────────────────────

# # Preprocessing (Part 2)
# from src.preprocessing.night_enhance import enhance_frame, is_low_light, adaptive_enhance

# # Lane detection (Part 2)
# from src.lane_detection.lane_detector import detect_lanes, draw_lanes
# from src.lane_detection.ego_corridor import compute_ego_corridor, filter_detections, draw_corridor

# # Weather (Part 2)
# from src.weather.weather_classifier import WeatherClassifier
# from src.weather.weather_to_physics import get_physics_params

# # Traffic signs (Part 3)
# from src.traffic_signs.sign_detector import TrafficSignDetector, adjust_aeb_threshold

# # Black box (Part 3)
# from src.recorder.black_box import BlackBoxRecorder

# # Collision (Part 1)
# from src.collision.collision_engine import compute_ttc, compute_distance_metric, compute_relative_speed
# from src.collision.aeb_controller import AEBController

# # Alertness (Part 5)
# from src.driver_monitoring.alertness_state import AlertnessState

# # ─── Optional heavy modules (may not be installed/available) ───────

# YOLO_AVAILABLE = False
# try:
#     from ultralytics import YOLO
#     YOLO_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: ultralytics not available. Detection disabled.")

# TRACKER_AVAILABLE = False
# try:
#     from src.tracking.tracker import Tracker
#     TRACKER_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: Tracker not available. Tracking disabled.")

# DEPTH_AVAILABLE = False
# try:
#     from src.depth.depth_estimator import DepthEstimator
#     DEPTH_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: DepthEstimator not available. Using pixel distance.")

# DYNAMICS_AVAILABLE = False
# try:
#     from src.vehicle_dynamics.dynamics_sim import simulate_braking, braking_distance
#     DYNAMICS_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: Vehicle dynamics not available.")

# IMPACT_AVAILABLE = False
# try:
#     from src.safety_analysis.impact_model import ImpactAnalyzer
#     IMPACT_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: Impact analysis not available.")

# DRIVER_MONITOR_AVAILABLE = False
# try:
#     from src.driver_monitoring.drowsiness_detector import DrowsinessDetector
#     DRIVER_MONITOR_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: Driver monitoring not available.")

# HEADPOSE_AVAILABLE = False
# try:
#     from src.driver_monitoring.head_pose import estimate_head_pose, is_distracted
#     HEADPOSE_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: Head pose estimation not available.")

# GRADCAM_AVAILABLE = False
# try:
#     from src.explainability.gradcam import GradCAM
#     GRADCAM_AVAILABLE = True
# except ImportError:
#     print("[Pipeline] WARNING: Grad-CAM not available.")


# # ═══════════════════════════════════════════════════════════════════
# # Pipeline Class
# # ═══════════════════════════════════════════════════════════════════

# class ADASPipeline:
#     """
#     Master ADAS pipeline orchestrating all perception, decision, and recording modules.
#     """

#     def __init__(self, args):
#         """
#         Initialize all sub-modules based on command-line arguments.

#         Args:
#             args: Namespace from argparse with video, speed, surface, etc.
#         """
#         self.args = args
#         self.speed_kmh = args.speed
#         self.speed_ms = args.speed / 3.6

#         # ── Output directory ──
#         self.output_dir = args.output
#         os.makedirs(self.output_dir, exist_ok=True)

#         # ── YOLO Detection ──
#         self.detector = None
#         if YOLO_AVAILABLE:
#             model_path = getattr(args, 'model', 'models/best.pt')
#             if os.path.exists(model_path):
#                 self.detector = YOLO(model_path)
#                 print(f"[Pipeline] YOLO loaded: {model_path}")
#             else:
#                 print(f"[Pipeline] YOLO model not found at {model_path}")

#         # ── Tracker ──
#         self.tracker = Tracker(max_age=3, min_hits=2, iou_threshold=0.3) if TRACKER_AVAILABLE else None

#         # ── Depth Estimator ──
#         self.depth_estimator = None
#         if DEPTH_AVAILABLE:
#             try:
#                 self.depth_estimator = DepthEstimator()
#             except Exception as e:
#                 print(f"[Pipeline] Depth estimator failed to load: {e}")

#         # ── Weather Classifier ──
#         self.weather_classifier = WeatherClassifier()

#         # ── Traffic Sign Detector ──
#         self.sign_detector = TrafficSignDetector()

#         # ── Driver Monitor ──
#         self.driver_monitor = None
#         if DRIVER_MONITOR_AVAILABLE:
#             try:
#                 self.driver_monitor = DrowsinessDetector()
#             except Exception as e:
#                 print(f"[Pipeline] Driver monitor failed to load: {e}")

#         # ── Impact Analyzer ──
#         self.impact_analyzer = ImpactAnalyzer() if IMPACT_AVAILABLE else None

#         # ── AEB Controller ──
#         self.aeb_controller = AEBController(ttc_threshold=1.5)

#         # ── Black Box Recorder ──
#         self.black_box = BlackBoxRecorder(buffer_seconds=30, fps=30)

#         # ── Grad-CAM ──
#         self.gradcam = None
#         if args.gradcam and GRADCAM_AVAILABLE:
#             model_path = getattr(args, 'model', 'models/best.pt')
#             if os.path.exists(model_path):
#                 self.gradcam = GradCAM(model_path)

#         # ── State tracking ──
#         self.frame_count = 0
#         self.aeb_trigger_count = 0
#         self.frame_logs = []
#         self.current_weather = "Clear"
#         self.current_surface = args.surface if args.surface != "auto" else "dry"

#     def run(self):
#         """Execute the full pipeline."""
#         # ── Open video source ──
#         if getattr(self.args, 'webcam', False):
#             cap = cv2.VideoCapture(0)
#             print("[Pipeline] Using webcam input")
#         else:
#             cap = cv2.VideoCapture(self.args.video)
#             print(f"[Pipeline] Processing: {self.args.video}")

#         if not cap.isOpened():
#             print("[Pipeline] ERROR: Cannot open video source!")
#             return

#         # ── Output video writer ──
#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#         self.black_box.fps = fps

#         output_video_path = os.path.join(self.output_dir, "output_video.mp4")
#         writer = cv2.VideoWriter(output_video_path,
#                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#         print(f"[Pipeline] Frame size: {w}x{h} @ {fps} FPS")
#         print(f"[Pipeline] Ego speed: {self.speed_kmh} km/h")
#         print(f"[Pipeline] Output: {self.output_dir}")
#         print("[Pipeline] Running...")

#         start_time = time.time()

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             self.frame_count += 1
#             annotated, frame_log = self._process_frame(frame)

#             # Write annotated frame
#             writer.write(annotated)

#             # Log this frame
#             self.frame_logs.append(frame_log)

#             # Progress report
#             if self.frame_count % 100 == 0:
#                 elapsed = time.time() - start_time
#                 fps_actual = self.frame_count / elapsed if elapsed > 0 else 0
#                 print(f"  Frame {self.frame_count} | {fps_actual:.1f} FPS | "
#                       f"Weather: {self.current_weather} | "
#                       f"AEB triggers: {self.aeb_trigger_count}")

#         cap.release()
#         writer.release()

#         elapsed = time.time() - start_time
#         print(f"\n[Pipeline] Complete: {self.frame_count} frames in {elapsed:.1f}s "
#               f"({self.frame_count / elapsed:.1f} FPS)")

#         # ── Save logs and summary ──
#         self._save_logs()
#         self._save_summary(elapsed)

#         # ── Generate Grad-CAM heatmaps for key frames ──
#         if self.gradcam and self.args.gradcam:
#             self._generate_gradcam_samples()

#     def _process_frame(self, frame):
#         """
#         Process a single frame through the entire ADAS pipeline.

#         Returns:
#             annotated: BGR frame with all overlays
#             frame_log: dict with all metrics for this frame
#         """
#         original = frame.copy()
#         frame_log = {
#             "frame": self.frame_count,
#             "timestamp": datetime.now().isoformat(),
#         }

#         # ── Step 1: CLAHE Enhancement (if low-light) ──
#         if self.args.clahe:
#             frame, was_enhanced = adaptive_enhance(frame)
#             frame_log["clahe_applied"] = was_enhanced

#         # ── Step 2: YOLO Detection ──
#         detections = []
#         det_array = np.empty((0, 6))

#         if self.detector is not None:
#             results = self.detector(frame, verbose=False)
#             for result in results:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                     conf = float(box.conf[0])
#                     cls_id = int(box.cls[0])
#                     cls_name = result.names.get(cls_id, "unknown")

#                     if conf > 0.3:
#                         detections.append({
#                             "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                             "confidence": conf,
#                             "class": cls_name,
#                             "class_id": cls_id,
#                         })

#             # Build detection array for tracker: [x1, y1, x2, y2, score, class_id]
#             if detections:
#                 det_array = np.array([
#                     [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
#                      d["confidence"], d["class_id"]]
#                     for d in detections
#                 ])

#         frame_log["num_detections"] = len(detections)

#         # ── Step 3: Multi-Object Tracking ──
#         tracked_objects = np.empty((0, 6))
#         if self.tracker is not None:
#             tracked_objects = self.tracker.update(det_array)

#         frame_log["num_tracked"] = len(tracked_objects)

#         # ── Step 4: Depth Estimation ──
#         depth_map = None
#         if self.depth_estimator is not None:
#             try:
#                 depth_map = self.depth_estimator.estimate_depth(frame)
#             except Exception:
#                 pass

#         # ── Step 5: Lane Detection ──
#         lanes = detect_lanes(frame)
#         left_lane = lanes[0] if len(lanes) >= 1 else None
#         right_lane = lanes[1] if len(lanes) >= 2 else None

#         # ── Step 6: Ego Corridor & In-Path Filtering ──
#         corridor = None
#         if left_lane and right_lane:
#             corridor = compute_ego_corridor(left_lane, right_lane, frame.shape[0])

#         in_path_dets, off_path_dets = filter_detections(detections, corridor)
#         frame_log["in_path_count"] = len(in_path_dets)
#         frame_log["filtered_out_count"] = len(off_path_dets)

#         # ── Step 7: Weather Classification ──
#         weather_result = self.weather_classifier.classify(frame)
#         self.current_weather = weather_result["weather"]
#         physics_params = get_physics_params(self.current_weather)

#         if self.args.surface == "auto":
#             self.current_surface = physics_params["surface"]

#         frame_log["weather"] = self.current_weather
#         frame_log["surface"] = self.current_surface

#         # ── Step 8: Traffic Sign Detection ──
#         speed_zone = self.sign_detector.get_active_speed_limit(frame)
#         adjusted_ttc_threshold = adjust_aeb_threshold(
#             self.aeb_controller.ttc_threshold, speed_zone["speed_limit_kmh"]
#         )
#         frame_log["speed_zone"] = speed_zone

#         # ── Step 9: Driver State (Drowsiness + Head Pose) ──
#         driver_state = AlertnessState.ALERT.value
#         driver_delay = 0.7  # default

#         if self.driver_monitor is not None:
#             try:
#                 _, state, delay = self.driver_monitor.process_frame(original)
#                 driver_state = state.value
#                 driver_delay = delay
#             except Exception:
#                 pass

#         total_reaction_time = driver_delay + physics_params["reaction_penalty"]
#         frame_log["driver_state"] = driver_state
#         frame_log["reaction_time"] = total_reaction_time

#         # ── Step 10: Per-Object Collision Analysis ──
#         min_ttc = float("inf")
#         aeb_triggered = False
#         closest_distance = float("inf")

#         for det in in_path_dets:
#             bbox = det["bbox"]

#             # Distance estimation
#             if depth_map is not None and self.depth_estimator is not None:
#                 distance_m = self.depth_estimator.get_distance_at_bbox(depth_map, bbox)
#             else:
#                 # Fallback: use bbox height as rough proxy
#                 bbox_height = bbox[3] - bbox[1]
#                 distance_m = max(1.0, 2000.0 / max(bbox_height, 1))

#             distance_m = compute_distance_metric(distance_m)

#             # Relative speed (assume pedestrian stationary for simplicity)
#             relative_speed = self.speed_ms

#             # Time to collision
#             ttc = compute_ttc(distance_m, relative_speed)

#             if ttc < min_ttc:
#                 min_ttc = ttc
#             if distance_m < closest_distance:
#                 closest_distance = distance_m

#             # AEB decision
#             effective_threshold = adjusted_ttc_threshold + total_reaction_time
#             if ttc < effective_threshold:
#                 aeb_triggered = True

#         # Update AEB controller state
#         if min_ttc < float("inf"):
#             self.aeb_controller.evaluate(min_ttc, reaction_time=total_reaction_time)

#         if aeb_triggered:
#             self.aeb_trigger_count += 1

#         frame_log["min_ttc"] = min_ttc if min_ttc < float("inf") else None
#         frame_log["closest_distance_m"] = closest_distance if closest_distance < float("inf") else None
#         frame_log["aeb_triggered"] = aeb_triggered

#         # ── Step 11: Impact Analysis (if AEB can't stop in time) ──
#         if aeb_triggered and DYNAMICS_AVAILABLE and self.impact_analyzer:
#             try:
#                 brake_dist = braking_distance(self.speed_kmh, self.current_surface)
#                 if closest_distance < brake_dist:
#                     residual_speed = self.impact_analyzer.compute_residual_speed(
#                         self.speed_kmh, brake_dist, closest_distance
#                     )
#                     impact_result = self.impact_analyzer.full_impact_analysis(residual_speed)
#                     frame_log["impact_analysis"] = {
#                         "residual_speed_kmh": residual_speed,
#                         "hic": impact_result["hic"],
#                         "severity": impact_result["description"],
#                     }
#             except Exception:
#                 pass

#         # ── Step 12: Black Box Recording ──
#         sensor_data = {
#             "timestamp": frame_log["timestamp"],
#             "frame_number": self.frame_count,
#             "ttc": min_ttc if min_ttc < float("inf") else None,
#             "distance_m": closest_distance if closest_distance < float("inf") else None,
#             "driver_state": driver_state,
#             "weather": self.current_weather,
#             "speed_kmh": self.speed_kmh,
#             "surface": self.current_surface,
#             "reaction_time": total_reaction_time,
#             "aeb_triggered": aeb_triggered,
#             "num_detections": len(detections),
#             "in_path_count": len(in_path_dets),
#         }
#         self.black_box.record_frame(frame, sensor_data)

#         # ── Step 13: Save Incident on AEB Trigger ──
#         if aeb_triggered:
#             incident_dir = os.path.join(self.output_dir, "incidents")
#             self.black_box.save_incident(incident_dir)

#         # ── Step 14: Annotate Frame ──
#         annotated = self._annotate_frame(
#             frame, detections, tracked_objects, lanes, corridor,
#             min_ttc, aeb_triggered, driver_state, total_reaction_time,
#             self.current_weather, speed_zone
#         )

#         return annotated, frame_log

#     def _annotate_frame(self, frame, detections, tracked_objects, lanes, corridor,
#                          ttc, aeb_triggered, driver_state, reaction_time,
#                          weather, speed_zone):
#         """Draw all overlays on the frame."""
#         annotated = frame.copy()

#         # Draw ego corridor
#         if corridor is not None:
#             annotated = draw_corridor(annotated, corridor, alpha=0.15)

#         # Draw lane lines
#         if lanes:
#             annotated = draw_lanes(annotated, lanes, color=(0, 255, 0), thickness=2)

#         # Draw detection bboxes
#         for det in detections:
#             x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
#             cls = det["class"]
#             conf = det["confidence"]
#             color = (0, 255, 0)  # Green default
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(annotated, f"{cls} {conf:.2f}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

#         # Draw tracked object IDs
#         for obj in tracked_objects:
#             x1, y1, x2, y2, track_id, cls_id = obj
#             cx, cy = int((x1 + x2) / 2), int(y1) - 10
#             cv2.putText(annotated, f"ID:{int(track_id)}", (cx, cy),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

#         # ── HUD Overlay (top-left info panel) ──
#         h, w = annotated.shape[:2]

#         # Semi-transparent background for HUD
#         overlay = annotated.copy()
#         cv2.rectangle(overlay, (5, 5), (320, 200), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

#         # TTC
#         ttc_text = f"TTC: {ttc:.2f}s" if ttc < float("inf") else "TTC: ---"
#         ttc_color = (0, 0, 255) if ttc < 3.0 else (0, 255, 255) if ttc < 5.0 else (0, 255, 0)
#         cv2.putText(annotated, ttc_text, (15, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, ttc_color, 2)

#         # AEB status
#         if aeb_triggered:
#             cv2.putText(annotated, "!! AEB TRIGGERED !!", (15, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             # Red border flash
#             cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
#         else:
#             cv2.putText(annotated, "AEB: Standby", (15, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

#         # Driver state
#         state_colors = {
#             "ALERT": (0, 255, 0), "TIRED": (0, 255, 255),
#             "DISTRACTED": (0, 165, 255), "DROWSY": (0, 128, 255),
#             "MICROSLEEP": (0, 0, 255),
#         }
#         state_color = state_colors.get(driver_state, (255, 255, 255))
#         cv2.putText(annotated, f"Driver: {driver_state}", (15, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

#         # Weather + Surface
#         cv2.putText(annotated, f"Weather: {weather} | Surface: {self.current_surface}",
#                     (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

#         # Speed zone
#         zone_text = f"Zone: {speed_zone['zone_type']} ({speed_zone['speed_limit_kmh']} km/h)"
#         cv2.putText(annotated, zone_text, (15, 140),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

#         # Reaction time
#         cv2.putText(annotated, f"Reaction: {reaction_time:.2f}s", (15, 165),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

#         # Speed
#         cv2.putText(annotated, f"Speed: {self.speed_kmh:.0f} km/h", (15, 190),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

#         return annotated

#     def _save_logs(self):
#         """Save per-frame pipeline logs to JSON."""
#         log_path = os.path.join(self.output_dir, "pipeline_log.json")

#         # Make serializable
#         serializable_logs = []
#         for log in self.frame_logs:
#             clean_log = {}
#             for k, v in log.items():
#                 if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
#                     clean_log[k] = str(v)
#                 elif isinstance(v, (np.floating,)):
#                     clean_log[k] = float(v)
#                 elif isinstance(v, (np.integer,)):
#                     clean_log[k] = int(v)
#                 else:
#                     clean_log[k] = v
#             serializable_logs.append(clean_log)

#         with open(log_path, "w") as f:
#             json.dump(serializable_logs, f, indent=2, default=str)
#         print(f"[Pipeline] Logs saved: {log_path}")

#     def _save_summary(self, elapsed_time):
#         """Save aggregate summary statistics."""
#         summary_path = os.path.join(self.output_dir, "pipeline_summary.json")

#         ttc_values = [l.get("min_ttc") for l in self.frame_logs
#                       if l.get("min_ttc") is not None]

#         summary = {
#             "run_timestamp": datetime.now().isoformat(),
#             "input_source": self.args.video if hasattr(self.args, 'video') else "webcam",
#             "total_frames": self.frame_count,
#             "processing_time_s": round(elapsed_time, 2),
#             "avg_fps": round(self.frame_count / elapsed_time, 1) if elapsed_time > 0 else 0,
#             "ego_speed_kmh": self.speed_kmh,
#             "surface": self.current_surface,
#             "aeb_trigger_count": self.aeb_trigger_count,
#             "min_ttc_observed": min(ttc_values) if ttc_values else None,
#             "mean_ttc": round(float(np.mean(ttc_values)), 2) if ttc_values else None,
#             "weather_detected": self.current_weather,
#             "modules_active": {
#                 "yolo": self.detector is not None,
#                 "tracker": self.tracker is not None,
#                 "depth": self.depth_estimator is not None,
#                 "driver_monitor": self.driver_monitor is not None,
#                 "gradcam": self.gradcam is not None,
#                 "dynamics": DYNAMICS_AVAILABLE,
#                 "impact": self.impact_analyzer is not None,
#             },
#         }

#         with open(summary_path, "w") as f:
#             json.dump(summary, f, indent=2, default=str)
#         print(f"[Pipeline] Summary saved: {summary_path}")

#         # Print to console
#         print("\n" + "=" * 60)
#         print("  ADAS Pipeline Summary")
#         print("=" * 60)
#         for k, v in summary.items():
#             if k != "modules_active":
#                 print(f"  {k}: {v}")
#         print(f"\n  Active modules:")
#         for mod, active in summary["modules_active"].items():
#             status = "✓" if active else "✗"
#             print(f"    {status} {mod}")
#         print("=" * 60)

#     def _generate_gradcam_samples(self):
#         """Generate Grad-CAM heatmaps for key frames (frames with AEB triggers)."""
#         if self.gradcam is None:
#             return

#         gradcam_dir = os.path.join(self.output_dir, "gradcam")
#         os.makedirs(gradcam_dir, exist_ok=True)

#         # Find frames where AEB was triggered
#         trigger_frames = [
#             i for i, log in enumerate(self.frame_logs)
#             if log.get("aeb_triggered")
#         ]

#         if not trigger_frames:
#             print("[Pipeline] No AEB triggers found for Grad-CAM analysis.")
#             return

#         # Process a sample of trigger frames (max 10)
#         sample = trigger_frames[:10]
#         print(f"[Pipeline] Generating Grad-CAM for {len(sample)} trigger frames...")

#         # Note: We'd need to re-read the video for this.
#         # For now, log the frame numbers for offline processing.
#         info = {"trigger_frames": sample, "total_triggers": len(trigger_frames)}
#         with open(os.path.join(gradcam_dir, "gradcam_info.json"), "w") as f:
#             json.dump(info, f, indent=2)

#         print(f"[Pipeline] Grad-CAM info saved to {gradcam_dir}")


# # ═══════════════════════════════════════════════════════════════════
# # Entry Point
# # ═══════════════════════════════════════════════════════════════════

# def run_pipeline(args):
#     """Create and run the ADAS pipeline."""
#     pipeline = ADASPipeline(args)
#     pipeline.run()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="ADAS Master Pipeline — End-to-End Pedestrian AEB System",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python run_pipeline.py --video data/sample/test_video.mp4 --speed 50
#   python run_pipeline.py --video driving.mp4 --speed 40 --surface wet
#   python run_pipeline.py --webcam --speed 30
#   python run_pipeline.py --video test.mp4 --speed 60 --gradcam
#         """
#     )

#     parser.add_argument("--video", type=str, help="Input video path")
#     parser.add_argument("--webcam", action="store_true", help="Use webcam input")
#     parser.add_argument("--model", type=str, default="models/best.pt",
#                         help="Path to YOLO model (default: models/best.pt)")
#     parser.add_argument("--speed", type=float, default=50.0,
#                         help="Ego vehicle speed in km/h (default: 50)")
#     parser.add_argument("--surface", type=str, default="auto",
#                         choices=["auto", "dry", "wet", "snow", "ice", "gravel"],
#                         help="Road surface (auto = from weather, default: auto)")
#     parser.add_argument("--output", type=str, default="results/pipeline/",
#                         help="Output directory (default: results/pipeline/)")
#     parser.add_argument("--clahe", action="store_true", default=True,
#                         help="Enable CLAHE auto-enhancement (default: True)")
#     parser.add_argument("--no-clahe", dest="clahe", action="store_false",
#                         help="Disable CLAHE enhancement")
#     parser.add_argument("--gradcam", action="store_true",
#                         help="Generate Grad-CAM heatmaps for AEB trigger frames")

#     args = parser.parse_args()

#     # Validate input
#     if not args.video and not args.webcam:
#         parser.error("Must provide --video or --webcam")

#     run_pipeline(args)


"""
ADAS Master Pipeline — runs the complete system end-to-end.

Integrates ALL modules from Parts 1–5 into a single per-frame processing loop:
    1. Preprocessing (CLAHE night vision)
    2. Detection (YOLO pedestrian/vehicle detection)
    3. Tracking (SORT multi-object tracking)
    4. Depth estimation (MiDaS monocular depth)
    5. Lane detection & ego corridor filtering
    6. Weather classification → physics parameters
    7. Traffic sign recognition → dynamic AEB thresholds
    8. Driver monitoring (drowsiness + head pose → reaction time)
    9. Collision prediction (TTC, vehicle dynamics, AEB decision)
    10. Impact analysis (HIC if collision unavoidable)
    11. Black box recording & incident preservation
    12. Frame annotation & output

Usage:
    python run_pipeline.py --source video --video data/sample/test_video.mp4 --speed 50
    python run_pipeline.py --source webcam --webcam-index 0 --speed 30
    python run_pipeline.py --source carla --speed 35 --carla-fps 20 --display
"""

import argparse
import cv2
import json
import os
import sys
import time
import numpy as np
from datetime import datetime

# ─── Add project root to path ─────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ─── Module imports (with graceful fallbacks) ──────────────────────

# Preprocessing
from src.preprocessing.night_enhance import enhance_frame, is_low_light, adaptive_enhance

# Lane detection
from src.lane_detection.lane_detector import detect_lanes, draw_lanes
from src.lane_detection.ego_corridor import compute_ego_corridor, filter_detections, draw_corridor

# Weather
from src.weather.weather_classifier import WeatherClassifier
from src.weather.weather_to_physics import get_physics_params

# Traffic signs
from src.traffic_signs.sign_detector import TrafficSignDetector, adjust_aeb_threshold

# Black box
from src.recorder.black_box import BlackBoxRecorder

# Collision
from src.collision.collision_engine import compute_ttc, compute_distance_metric, compute_relative_speed
from src.collision.aeb_controller import AEBController

# Alertness
from src.driver_monitoring.alertness_state import AlertnessState

# ─── Optional heavy modules ────────────────────────────────────────

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: ultralytics not available. Detection disabled.")

TRACKER_AVAILABLE = False
try:
    from src.tracking.tracker import Tracker
    TRACKER_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: Tracker not available. Tracking disabled.")

DEPTH_AVAILABLE = False
try:
    from src.depth.depth_estimator import DepthEstimator
    DEPTH_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: DepthEstimator not available. Using pixel distance.")

DYNAMICS_AVAILABLE = False
try:
    from src.vehicle_dynamics.dynamics_sim import simulate_braking, braking_distance
    DYNAMICS_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: Vehicle dynamics not available.")

IMPACT_AVAILABLE = False
try:
    from src.safety_analysis.impact_model import ImpactAnalyzer
    IMPACT_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: Impact analysis not available.")

DRIVER_MONITOR_AVAILABLE = False
try:
    from src.driver_monitoring.drowsiness_detector import DrowsinessDetector
    DRIVER_MONITOR_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: Driver monitoring not available.")

HEADPOSE_AVAILABLE = False
try:
    from src.driver_monitoring.head_pose import estimate_head_pose, is_distracted
    HEADPOSE_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: Head pose estimation not available.")

GRADCAM_AVAILABLE = False
try:
    from src.explainability.gradcam import GradCAM
    GRADCAM_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: Grad-CAM not available.")

CARLA_AVAILABLE = False
try:
    from src.carla_sim.carla_bridge import CARLABridge
    from src.carla_sim.carla_scenarios import ScenarioManager
    CARLA_AVAILABLE = True
except ImportError:
    print("[Pipeline] WARNING: CARLA bridge/scenarios not available. CARLA mode disabled.")


# ═══════════════════════════════════════════════════════════════════
# Pipeline Class
# ═══════════════════════════════════════════════════════════════════

class ADASPipeline:
    """
    Master ADAS pipeline orchestrating all perception, decision, and recording modules.
    """

    def __init__(self, args):
        """
        Initialize all sub-modules based on command-line arguments.

        Args:
            args: Namespace from argparse with input, speed, surface, etc.
        """
        self.args = args
        self.source_mode = args.source
        self.speed_kmh = args.speed
        self.speed_ms = args.speed / 3.6

        # ── Output directory ──
        self.output_dir = args.output
        os.makedirs(self.output_dir, exist_ok=True)

        # ── YOLO Detection ──
        self.detector = None
        if YOLO_AVAILABLE:
            model_path = getattr(args, "model", "models/best.pt")
            if os.path.exists(model_path):
                self.detector = YOLO(model_path)
                print(f"[Pipeline] YOLO loaded: {model_path}")
            else:
                print(f"[Pipeline] YOLO model not found at {model_path}")

        # ── Tracker ──
        self.tracker = Tracker(max_age=3, min_hits=2, iou_threshold=0.3) if TRACKER_AVAILABLE else None

        # ── Depth Estimator ──
        self.depth_estimator = None
        if DEPTH_AVAILABLE:
            try:
                self.depth_estimator = DepthEstimator()
            except Exception as e:
                print(f"[Pipeline] Depth estimator failed to load: {e}")

        # ── Weather Classifier ──
        self.weather_classifier = WeatherClassifier()

        # ── Traffic Sign Detector ──
        self.sign_detector = TrafficSignDetector()

        # ── Driver Monitor ──
        self.driver_monitor = None
        if DRIVER_MONITOR_AVAILABLE:
            try:
                self.driver_monitor = DrowsinessDetector()
            except Exception as e:
                print(f"[Pipeline] Driver monitor failed to load: {e}")

        # ── Impact Analyzer ──
        self.impact_analyzer = ImpactAnalyzer() if IMPACT_AVAILABLE else None

        # ── AEB Controller ──
        self.aeb_controller = AEBController(ttc_threshold=1.5)

        # ── Black Box Recorder ──
        self.black_box = BlackBoxRecorder(buffer_seconds=30, fps=30)

        # ── Grad-CAM ──
        self.gradcam = None
        if args.gradcam and GRADCAM_AVAILABLE:
            model_path = getattr(args, "model", "models/best.pt")
            if os.path.exists(model_path):
                self.gradcam = GradCAM(model_path)

        # ── State tracking ──
        self.frame_count = 0
        self.aeb_trigger_count = 0
        self.frame_logs = []
        self.current_weather = "Clear"
        self.current_surface = args.surface if args.surface != "auto" else "dry"

    def run(self):
        """Execute the full pipeline for the selected source."""
        if self.source_mode == "video":
            self._run_video_source()
        elif self.source_mode == "webcam":
            self._run_webcam_source()
        elif self.source_mode == "carla":
            self._run_carla_source()
        else:
            raise ValueError(f"Unsupported source mode: {self.source_mode}")

    def _run_capture_loop(self, cap, source_name="video"):
        """Shared processing loop for OpenCV capture sources."""
        if not cap.isOpened():
            print(f"[Pipeline] ERROR: Cannot open {source_name} source!")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        self.black_box.fps = fps

        output_video_path = os.path.join(self.output_dir, "output_video.mp4")
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        print(f"[Pipeline] Source: {source_name}")
        print(f"[Pipeline] Frame size: {w}x{h} @ {fps} FPS")
        print(f"[Pipeline] Ego speed: {self.speed_kmh} km/h")
        print(f"[Pipeline] Output: {self.output_dir}")
        print("[Pipeline] Running...")

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            annotated, frame_log = self._process_frame(frame)
            writer.write(annotated)
            self.frame_logs.append(frame_log)

            if self.args.display:
                cv2.imshow("ADAS Pipeline", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if self.args.max_frames and self.frame_count >= self.args.max_frames:
                break

            if self.frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {self.frame_count} | {fps_actual:.1f} FPS | "
                      f"Weather: {self.current_weather} | "
                      f"AEB triggers: {self.aeb_trigger_count}")

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n[Pipeline] Complete: {self.frame_count} frames in {elapsed:.1f}s "
              f"({self.frame_count / elapsed:.1f} FPS)")

        self._save_logs()
        self._save_summary(elapsed)

        if self.gradcam and self.args.gradcam:
            self._generate_gradcam_samples()

    def _run_video_source(self):
        if not self.args.video:
            raise ValueError("Video mode selected but no --video path provided")
        print(f"[Pipeline] Processing video: {self.args.video}")
        cap = cv2.VideoCapture(self.args.video)
        self._run_capture_loop(cap, source_name=self.args.video)

    def _run_webcam_source(self):
        print(f"[Pipeline] Using webcam index {self.args.webcam_index}")
        cap = cv2.VideoCapture(self.args.webcam_index)
        self._run_capture_loop(cap, source_name=f"webcam:{self.args.webcam_index}")

    def _run_carla_source(self):
        if not CARLA_AVAILABLE:
            print("[Pipeline] ERROR: CARLA mode requested but CARLA bridge/scenario manager is unavailable.")
            return

        bridge = CARLABridge(
            host=self.args.carla_host,
            port=self.args.carla_port,
            fps=self.args.carla_fps
        )

        writer = None
        scenario_manager = None
        start_time = time.time()

        try:
            print("[Pipeline] Connecting to CARLA...")
            bridge.spawn_ego_vehicle(vehicle_type=self.args.carla_vehicle_type)
            bridge.attach_camera(
                width=self.args.carla_width,
                height=self.args.carla_height,
                fov=self.args.carla_fov
            )

            if self.args.carla_weather:
                bridge.set_weather(self.args.carla_weather)

            scenario_manager = ScenarioManager(bridge.world, bridge.ego_vehicle)

            print(f"[Pipeline] CARLA scenario: {self.args.carla_scenario}")
            print(f"[Pipeline] Pedestrian speed: {self.args.pedestrian_speed} m/s")
            print(f"[Pipeline] Pedestrian offset: {self.args.pedestrian_offset} m")

            if self.args.carla_scenario == "crossing":
                scenario_manager.spawn_crossing_pedestrian(
                    crossing_speed=self.args.pedestrian_speed,
                    start_offset=self.args.pedestrian_offset
                )
            elif self.args.carla_scenario == "standing":
                scenario_manager.spawn_standing_pedestrian(
                    distance=self.args.pedestrian_offset
                )
            elif self.args.carla_scenario == "running":
                scenario_manager.spawn_running_pedestrian(
                    start_offset=self.args.pedestrian_offset
                )
            elif self.args.carla_scenario == "child":
                scenario_manager.spawn_child_behind_car()
            elif self.args.carla_scenario == "school_zone":
                scenario_manager.setup_school_zone()
            elif self.args.carla_scenario == "none":
                scenario_manager = None

            self.black_box.fps = self.args.carla_fps

            output_video_path = os.path.join(self.output_dir, "output_video.mp4")
            writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.args.carla_fps,
                (self.args.carla_width, self.args.carla_height)
            )

            print(f"[Pipeline] CARLA mode running @ {self.args.carla_fps} FPS")
            print(f"[Pipeline] Resolution: {self.args.carla_width}x{self.args.carla_height}")
            print(f"[Pipeline] Ego speed target: {self.speed_kmh} km/h")
            print(f"[Pipeline] Output: {self.output_dir}")

            while True:
                bridge.tick()
                frame = bridge.get_frame()
                self.frame_count += 1

                carla_speed = bridge.get_vehicle_speed()
                self.speed_kmh = carla_speed
                self.speed_ms = carla_speed / 3.6

                annotated, frame_log = self._process_frame(frame)

                brake_cmd = 1.0 if frame_log.get("aeb_triggered", False) else 0.0
                throttle_cmd = 0.0 if brake_cmd > 0 else self.args.carla_throttle

                if brake_cmd > 0:
                    bridge.apply_brake(brake_cmd)
                else:
                    bridge.apply_throttle(throttle_cmd)

                frame_log["source"] = "carla"
                frame_log["carla_speed_kmh"] = carla_speed
                frame_log["carla_brake_cmd"] = brake_cmd
                frame_log["carla_throttle_cmd"] = throttle_cmd
                frame_log["carla_scenario"] = self.args.carla_scenario

                self.frame_logs.append(frame_log)
                writer.write(annotated)

                if self.args.display:
                    cv2.imshow("ADAS Pipeline - CARLA", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break

                if self.args.max_frames and self.frame_count >= self.args.max_frames:
                    break

                if self.frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"  Frame {self.frame_count} | {fps_actual:.1f} FPS | "
                          f"Speed: {carla_speed:.1f} km/h | "
                          f"AEB triggers: {self.aeb_trigger_count}")

        finally:
            if writer is not None:
                writer.release()
            if scenario_manager is not None:
                scenario_manager.cleanup()
            bridge.cleanup()
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n[Pipeline] Complete: {self.frame_count} frames in {elapsed:.1f}s "
              f"({self.frame_count / elapsed:.1f} FPS)")

        self._save_logs()
        self._save_summary(elapsed)

        if self.gradcam and self.args.gradcam:
            self._generate_gradcam_samples()

    def _process_frame(self, frame):
        """
        Process a single frame through the entire ADAS pipeline.

        Returns:
            annotated: BGR frame with all overlays
            frame_log: dict with all metrics for this frame
        """
        original = frame.copy()
        frame_log = {
            "frame": self.frame_count,
            "timestamp": datetime.now().isoformat(),
        }

        # ── Step 1: CLAHE Enhancement ──
        if self.args.clahe:
            frame, was_enhanced = adaptive_enhance(frame)
            frame_log["clahe_applied"] = was_enhanced

        # ── Step 2: YOLO Detection ──
        detections = []
        det_array = np.empty((0, 6))

        if self.detector is not None:
            results = self.detector(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names.get(cls_id, "unknown")

                    if conf > 0.3:
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class": cls_name,
                            "class_id": cls_id,
                        })

            if detections:
                det_array = np.array([
                    [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                     d["confidence"], d["class_id"]]
                    for d in detections
                ])

        frame_log["num_detections"] = len(detections)

        # ── Step 3: Tracking ──
        tracked_objects = np.empty((0, 6))
        if self.tracker is not None:
            tracked_objects = self.tracker.update(det_array)

        frame_log["num_tracked"] = len(tracked_objects)

        # ── Step 4: Depth ──
        depth_map = None
        if self.depth_estimator is not None:
            try:
                depth_map = self.depth_estimator.estimate_depth(frame)
            except Exception:
                pass

        # ── Step 5: Lane Detection ──
        lanes = detect_lanes(frame)
        left_lane = lanes[0] if len(lanes) >= 1 else None
        right_lane = lanes[1] if len(lanes) >= 2 else None

        # ── Step 6: Corridor Filtering ──
        corridor = None
        if left_lane and right_lane:
            corridor = compute_ego_corridor(left_lane, right_lane, frame.shape[0])

        in_path_dets, off_path_dets = filter_detections(detections, corridor)
        frame_log["in_path_count"] = len(in_path_dets)
        frame_log["filtered_out_count"] = len(off_path_dets)

        # ── Step 7: Weather ──
        weather_result = self.weather_classifier.classify(frame)
        self.current_weather = weather_result["weather"]
        physics_params = get_physics_params(self.current_weather)

        if self.args.surface == "auto":
            self.current_surface = physics_params["surface"]

        frame_log["weather"] = self.current_weather
        frame_log["surface"] = self.current_surface

        # ── Step 8: Traffic Sign Detection ──
        speed_zone = self.sign_detector.get_active_speed_limit(frame)
        adjusted_ttc_threshold = adjust_aeb_threshold(
            self.aeb_controller.ttc_threshold, speed_zone["speed_limit_kmh"]
        )
        frame_log["speed_zone"] = speed_zone

        # ── Step 9: Driver State ──
        driver_state = AlertnessState.ALERT.value
        driver_delay = 0.7

        if self.driver_monitor is not None:
            try:
                _, state, delay = self.driver_monitor.process_frame(original)
                driver_state = state.value
                driver_delay = delay
            except Exception:
                pass

        total_reaction_time = driver_delay + physics_params["reaction_penalty"]
        frame_log["driver_state"] = driver_state
        frame_log["reaction_time"] = total_reaction_time

        # ── Step 10: Collision Analysis ──
        min_ttc = float("inf")
        aeb_triggered = False
        closest_distance = float("inf")

        for det in in_path_dets:
            bbox = det["bbox"]

            if depth_map is not None and self.depth_estimator is not None:
                distance_m = self.depth_estimator.get_distance_at_bbox(depth_map, bbox)
            else:
                bbox_height = bbox[3] - bbox[1]
                distance_m = max(1.0, 2000.0 / max(bbox_height, 1))

            distance_m = compute_distance_metric(distance_m)
            relative_speed = self.speed_ms
            ttc = compute_ttc(distance_m, relative_speed)

            if ttc < min_ttc:
                min_ttc = ttc
            if distance_m < closest_distance:
                closest_distance = distance_m

            effective_threshold = adjusted_ttc_threshold + total_reaction_time
            if ttc < effective_threshold:
                aeb_triggered = True

        if min_ttc < float("inf"):
            self.aeb_controller.evaluate(min_ttc, reaction_time=total_reaction_time)

        if aeb_triggered:
            self.aeb_trigger_count += 1

        frame_log["min_ttc"] = min_ttc if min_ttc < float("inf") else None
        frame_log["closest_distance_m"] = closest_distance if closest_distance < float("inf") else None
        frame_log["aeb_triggered"] = aeb_triggered

        # ── Step 11: Impact Analysis ──
        if aeb_triggered and DYNAMICS_AVAILABLE and self.impact_analyzer:
            try:
                brake_dist = braking_distance(self.speed_kmh, self.current_surface)
                if closest_distance < brake_dist:
                    residual_speed = self.impact_analyzer.compute_residual_speed(
                        self.speed_kmh, brake_dist, closest_distance
                    )
                    impact_result = self.impact_analyzer.full_impact_analysis(residual_speed)
                    frame_log["impact_analysis"] = {
                        "residual_speed_kmh": residual_speed,
                        "hic": impact_result["hic"],
                        "severity": impact_result["description"],
                    }
            except Exception:
                pass

        # ── Step 12: Black Box Recording ──
        sensor_data = {
            "timestamp": frame_log["timestamp"],
            "frame_number": self.frame_count,
            "ttc": min_ttc if min_ttc < float("inf") else None,
            "distance_m": closest_distance if closest_distance < float("inf") else None,
            "driver_state": driver_state,
            "weather": self.current_weather,
            "speed_kmh": self.speed_kmh,
            "surface": self.current_surface,
            "reaction_time": total_reaction_time,
            "aeb_triggered": aeb_triggered,
            "num_detections": len(detections),
            "in_path_count": len(in_path_dets),
        }
        self.black_box.record_frame(frame, sensor_data)

        # ── Step 13: Save Incident ──
        if aeb_triggered:
            incident_dir = os.path.join(self.output_dir, "incidents")
            self.black_box.save_incident(incident_dir)

        # ── Step 14: Annotate ──
        annotated = self._annotate_frame(
            frame, detections, tracked_objects, lanes, corridor,
            min_ttc, aeb_triggered, driver_state, total_reaction_time,
            self.current_weather, speed_zone
        )

        return annotated, frame_log

    def _annotate_frame(self, frame, detections, tracked_objects, lanes, corridor,
                         ttc, aeb_triggered, driver_state, reaction_time,
                         weather, speed_zone):
        """Draw all overlays on the frame."""
        annotated = frame.copy()

        if corridor is not None:
            annotated = draw_corridor(annotated, corridor, alpha=0.15)

        if lanes:
            annotated = draw_lanes(annotated, lanes, color=(0, 255, 0), thickness=2)

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            cls = det["class"]
            conf = det["confidence"]
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, cls_id = obj
            cx, cy = int((x1 + x2) / 2), int(y1) - 10
            cv2.putText(annotated, f"ID:{int(track_id)}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (320, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        ttc_text = f"TTC: {ttc:.2f}s" if ttc < float("inf") else "TTC: ---"
        ttc_color = (0, 0, 255) if ttc < 3.0 else (0, 255, 255) if ttc < 5.0 else (0, 255, 0)
        cv2.putText(annotated, ttc_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ttc_color, 2)

        if aeb_triggered:
            cv2.putText(annotated, "!! AEB TRIGGERED !!", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        else:
            cv2.putText(annotated, "AEB: Standby", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        state_colors = {
            "ALERT": (0, 255, 0),
            "TIRED": (0, 255, 255),
            "DISTRACTED": (0, 165, 255),
            "DROWSY": (0, 128, 255),
            "MICROSLEEP": (0, 0, 255),
        }
        state_color = state_colors.get(driver_state, (255, 255, 255))
        cv2.putText(annotated, f"Driver: {driver_state}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

        cv2.putText(annotated, f"Weather: {weather} | Surface: {self.current_surface}",
                    (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        zone_text = f"Zone: {speed_zone['zone_type']} ({speed_zone['speed_limit_kmh']} km/h)"
        cv2.putText(annotated, zone_text, (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(annotated, f"Reaction: {reaction_time:.2f}s", (15, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(annotated, f"Speed: {self.speed_kmh:.1f} km/h", (15, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return annotated

    def _save_logs(self):
        """Save per-frame pipeline logs to JSON."""
        log_path = os.path.join(self.output_dir, "pipeline_log.json")

        serializable_logs = []
        for log in self.frame_logs:
            clean_log = {}
            for k, v in log.items():
                if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
                    clean_log[k] = str(v)
                elif isinstance(v, (np.floating,)):
                    clean_log[k] = float(v)
                elif isinstance(v, (np.integer,)):
                    clean_log[k] = int(v)
                else:
                    clean_log[k] = v
            serializable_logs.append(clean_log)

        with open(log_path, "w") as f:
            json.dump(serializable_logs, f, indent=2, default=str)
        print(f"[Pipeline] Logs saved: {log_path}")

    def _save_summary(self, elapsed_time):
        """Save aggregate summary statistics."""
        summary_path = os.path.join(self.output_dir, "pipeline_summary.json")

        ttc_values = [l.get("min_ttc") for l in self.frame_logs if l.get("min_ttc") is not None]

        input_source = (
            self.args.video if self.source_mode == "video"
            else f"webcam:{self.args.webcam_index}" if self.source_mode == "webcam"
            else f"carla:{self.args.carla_host}:{self.args.carla_port}"
        )

        summary = {
            "run_timestamp": datetime.now().isoformat(),
            "input_source": input_source,
            "source_mode": self.source_mode,
            "total_frames": self.frame_count,
            "processing_time_s": round(elapsed_time, 2),
            "avg_fps": round(self.frame_count / elapsed_time, 1) if elapsed_time > 0 else 0,
            "ego_speed_kmh": self.speed_kmh,
            "surface": self.current_surface,
            "aeb_trigger_count": self.aeb_trigger_count,
            "min_ttc_observed": min(ttc_values) if ttc_values else None,
            "mean_ttc": round(float(np.mean(ttc_values)), 2) if ttc_values else None,
            "weather_detected": self.current_weather,
            "modules_active": {
                "yolo": self.detector is not None,
                "tracker": self.tracker is not None,
                "depth": self.depth_estimator is not None,
                "driver_monitor": self.driver_monitor is not None,
                "gradcam": self.gradcam is not None,
                "dynamics": DYNAMICS_AVAILABLE,
                "impact": self.impact_analyzer is not None,
                "carla": CARLA_AVAILABLE,
            },
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[Pipeline] Summary saved: {summary_path}")

        print("\n" + "=" * 60)
        print("  ADAS Pipeline Summary")
        print("=" * 60)
        for k, v in summary.items():
            if k != "modules_active":
                print(f"  {k}: {v}")
        print("\n  Active modules:")
        for mod, active in summary["modules_active"].items():
            status = "✓" if active else "✗"
            print(f"    {status} {mod}")
        print("=" * 60)

    def _generate_gradcam_samples(self):
        """Generate Grad-CAM heatmaps for key frames."""
        if self.gradcam is None:
            return

        gradcam_dir = os.path.join(self.output_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        trigger_frames = [
            i for i, log in enumerate(self.frame_logs)
            if log.get("aeb_triggered")
        ]

        if not trigger_frames:
            print("[Pipeline] No AEB triggers found for Grad-CAM analysis.")
            return

        sample = trigger_frames[:10]
        print(f"[Pipeline] Generating Grad-CAM info for {len(sample)} trigger frames...")

        info = {"trigger_frames": sample, "total_triggers": len(trigger_frames)}
        with open(os.path.join(gradcam_dir, "gradcam_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"[Pipeline] Grad-CAM info saved to {gradcam_dir}")


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Create and run the ADAS pipeline."""
    pipeline = ADASPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADAS Master Pipeline — End-to-End Pedestrian AEB System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --source video --video data/sample/test_video.mp4 --speed 50
  python run_pipeline.py --source webcam --webcam-index 0 --speed 30
  python run_pipeline.py --source video --video driving.mp4 --speed 40 --surface wet
  python run_pipeline.py --source carla --speed 35 --carla-fps 20 --display
  python run_pipeline.py --source carla --speed 30 --carla-width 640 --carla-height 480
  python run_pipeline.py --source carla --carla-scenario crossing --pedestrian-speed 1.4 --pedestrian-offset 12
        """
    )

    # Input source selection
    parser.add_argument("--source", type=str, default="video",
                        choices=["video", "webcam", "carla"],
                        help="Input source mode")

    # Video / webcam
    parser.add_argument("--video", type=str, help="Input video path for --source video")
    parser.add_argument("--webcam-index", type=int, default=0,
                        help="Webcam index for --source webcam")

    # General options
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to YOLO model (default: models/best.pt)")
    parser.add_argument("--speed", type=float, default=50.0,
                        help="Initial ego vehicle speed in km/h (default: 50)")
    parser.add_argument("--surface", type=str, default="auto",
                        choices=["auto", "dry", "wet", "snow", "ice", "gravel"],
                        help="Road surface (auto = from weather, default: auto)")
    parser.add_argument("--output", type=str, default="results/pipeline/",
                        help="Output directory")
    parser.add_argument("--display", action="store_true",
                        help="Show live annotated output window")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = unlimited)")
    parser.add_argument("--clahe", action="store_true", default=True,
                        help="Enable CLAHE auto-enhancement (default: True)")
    parser.add_argument("--no-clahe", dest="clahe", action="store_false",
                        help="Disable CLAHE enhancement")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM heatmaps for AEB trigger frames")

    # CARLA options
    parser.add_argument("--carla-host", type=str, default="localhost",
                        help="CARLA server host")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA server port")
    parser.add_argument("--carla-fps", type=int, default=20,
                        help="CARLA synchronous simulation FPS")
    parser.add_argument("--carla-width", type=int, default=640,
                        help="CARLA camera width")
    parser.add_argument("--carla-height", type=int, default=480,
                        help="CARLA camera height")
    parser.add_argument("--carla-fov", type=float, default=90.0,
                        help="CARLA RGB camera FOV")
    parser.add_argument("--carla-throttle", type=float, default=0.35,
                        help="Throttle to apply when AEB is not braking")
    parser.add_argument("--carla-vehicle-type", type=str, default="vehicle.tesla.model3",
                        help="CARLA blueprint id for ego vehicle")
    parser.add_argument("--carla-weather", type=str, default=None,
                        choices=[None, "Clear", "Rain", "Fog", "Night"],
                        help="Optional CARLA weather preset")

    # CARLA scenario options
    parser.add_argument("--carla-scenario", type=str, default="crossing",
                        choices=["none", "crossing", "standing", "running", "child", "school_zone"],
                        help="CARLA pedestrian scenario")
    parser.add_argument("--pedestrian-speed", type=float, default=1.4,
                        help="Pedestrian speed in m/s for supported scenarios")
    parser.add_argument("--pedestrian-offset", type=float, default=12.0,
                        help="Distance ahead of ego vehicle for scenario spawn")

    args = parser.parse_args()

    # Validation
    if args.source == "video" and not args.video:
        parser.error("--source video requires --video")
    if args.source == "carla" and not CARLA_AVAILABLE:
        parser.error("CARLA mode requested but src.carla_sim modules are unavailable")

    run_pipeline(args)