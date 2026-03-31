"""
Black Box Incident Recorder (Event Data Recorder).

Maintains a circular buffer of the last N seconds of video frames and sensor
data. When AEB triggers, the buffer is frozen and saved as an incident package
containing video, sensor logs, summary, and diagnostic plots.

Like a car's EDR (Event Data Recorder) or a dashcam's loop recording,
but enriched with all ADAS sensor data for post-incident analysis.

Incident package structure:
    incidents/
    └── incident_YYYY-MM-DD_HH-MM-SS/
        ├── video.mp4              (last 30 seconds of footage)
        ├── sensor_log.json        (all sensor data per frame)
        ├── summary.json           (key metrics at trigger moment)
        └── ttc_timeline.png       (TTC over the 30s buffer)
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from collections import deque

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/CI
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BlackBoxRecorder:
    """
    Circular buffer Event Data Recorder.

    Records the last N seconds of video + sensor data.
    On AEB trigger → saves everything as an incident evidence package.
    """

    def __init__(self, buffer_seconds=30, fps=30):
        """
        Initialize circular buffer.

        Args:
            buffer_seconds: How many seconds of history to keep (default 30)
            fps: Expected frame rate (used to size the buffer)
        """
        self.buffer_size = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.incident_count = 0

    def record_frame(self, frame, sensor_data):
        """
        Add one frame and its sensor data to the circular buffer.

        The buffer automatically discards the oldest entries when full,
        maintaining a rolling window of the most recent frames.

        Args:
            frame: BGR frame (numpy H×W×3). A copy is stored to prevent
                   external mutation.
            sensor_data: dict containing any/all of:
                "timestamp": str (ISO format)
                "frame_number": int
                "ttc": float (time-to-collision in seconds)
                "distance_m": float (distance to nearest pedestrian)
                "driver_state": str (ALERT/TIRED/DROWSY/DISTRACTED/MICROSLEEP)
                "weather": str (Clear/Rain/Fog/Night)
                "speed_kmh": float (ego vehicle speed)
                "aeb_triggered": bool
                "detections": list of detection dicts
                "surface": str (road surface type)
                "reaction_time": float (total reaction time)
        """
        self.frame_buffer.append(frame.copy())

        # Ensure sensor_data has a timestamp
        data = dict(sensor_data)  # shallow copy
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()

        self.data_buffer.append(data)

    def save_incident(self, output_dir="incidents"):
        """
        Freeze the buffer and save as an incident evidence package.

        Creates a timestamped folder with:
        - video.mp4: reconstructed video from frame buffer
        - sensor_log.json: all sensor data for each buffered frame
        - summary.json: key metrics at the moment of AEB trigger
        - ttc_timeline.png: plot of TTC values over the buffer window

        Args:
            output_dir: Base directory for incident packages

        Returns:
            incident_dir: str — path to the saved incident package
        """
        if len(self.frame_buffer) == 0:
            print("[BlackBox] No frames in buffer, nothing to save.")
            return None

        self.incident_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        incident_dir = os.path.join(output_dir,
                                     f"incident_{timestamp}_{self.incident_count:03d}")
        os.makedirs(incident_dir, exist_ok=True)

        print(f"[BlackBox] Saving incident package to {incident_dir}...")

        # Save all components
        self._save_video(incident_dir)
        self._save_sensor_log(incident_dir)
        self._save_summary(incident_dir)
        self._plot_ttc_timeline(incident_dir)

        print(f"[BlackBox] Incident package saved: {incident_dir}")
        return incident_dir

    def _save_video(self, incident_dir):
        """Write frame buffer to MP4 video."""
        if len(self.frame_buffer) == 0:
            return

        video_path = os.path.join(incident_dir, "video.mp4")
        first_frame = self.frame_buffer[0]
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

        for frame in self.frame_buffer:
            writer.write(frame)

        writer.release()
        print(f"  → Video: {len(self.frame_buffer)} frames saved")

    def _save_sensor_log(self, incident_dir):
        """Write all buffered sensor data to JSON."""
        log_path = os.path.join(incident_dir, "sensor_log.json")

        log_data = []
        for data in self.data_buffer:
            # Ensure everything is JSON-serializable
            serializable = self._make_serializable(data)
            log_data.append(serializable)

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        print(f"  → Sensor log: {len(log_data)} entries saved")

    def _save_summary(self, incident_dir):
        """Extract key info from the moment AEB triggered."""
        summary_path = os.path.join(incident_dir, "summary.json")

        if len(self.data_buffer) == 0:
            summary = {"error": "No sensor data available"}
        else:
            # The trigger moment is the last entry in the buffer
            trigger_data = dict(self.data_buffer[-1])

            # Compute some aggregate stats over the buffer
            ttc_values = [d.get("ttc", float("inf")) for d in self.data_buffer
                          if d.get("ttc") is not None and d.get("ttc") != float("inf")]

            summary = {
                "incident_timestamp": datetime.now().isoformat(),
                "buffer_duration_s": len(self.frame_buffer) / max(self.fps, 1),
                "total_frames": len(self.frame_buffer),
                "trigger_frame_data": self._make_serializable(trigger_data),
                "stats": {
                    "min_ttc": min(ttc_values) if ttc_values else None,
                    "mean_ttc": float(np.mean(ttc_values)) if ttc_values else None,
                    "frames_with_aeb": sum(
                        1 for d in self.data_buffer if d.get("aeb_triggered")
                    ),
                },
            }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  → Summary saved")

    def _plot_ttc_timeline(self, incident_dir):
        """Plot TTC values over the buffer window. Mark the trigger point."""
        if not MATPLOTLIB_AVAILABLE:
            print("  → TTC plot skipped (matplotlib not available)")
            return

        plot_path = os.path.join(incident_dir, "ttc_timeline.png")

        ttc_values = []
        time_axis = []
        aeb_triggers = []

        for i, data in enumerate(self.data_buffer):
            t = i / max(self.fps, 1)  # seconds from buffer start
            time_axis.append(t)

            ttc = data.get("ttc")
            if ttc is not None and ttc != float("inf") and ttc < 100:
                ttc_values.append(ttc)
            else:
                ttc_values.append(None)

            if data.get("aeb_triggered"):
                aeb_triggers.append(t)

        # Filter out None values for plotting
        valid_times = [t for t, v in zip(time_axis, ttc_values) if v is not None]
        valid_ttcs = [v for v in ttc_values if v is not None]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        if valid_times:
            ax.plot(valid_times, valid_ttcs, color='#00d4ff', linewidth=2,
                    label='TTC (seconds)', alpha=0.9)
            ax.fill_between(valid_times, valid_ttcs, alpha=0.15, color='#00d4ff')

        # Mark AEB trigger points
        for t in aeb_triggers:
            ax.axvline(x=t, color='#ff3333', linestyle='--', alpha=0.8,
                       linewidth=1.5, label='AEB Trigger' if t == aeb_triggers[0] else None)

        # Danger zone
        ax.axhline(y=1.5, color='#ff6600', linestyle=':', alpha=0.5,
                   label='TTC Threshold (1.5s)')
        ax.fill_between(time_axis, 0, 1.5, alpha=0.08, color='red')

        ax.set_xlabel('Time (seconds)', color='white', fontsize=11)
        ax.set_ylabel('TTC (seconds)', color='white', fontsize=11)
        ax.set_title('Time-to-Collision Timeline — Incident Buffer',
                     color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
        ax.grid(True, alpha=0.15, color='white')

        for spine in ax.spines.values():
            spine.set_color('#444')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"  → TTC timeline plot saved")

    @staticmethod
    def _make_serializable(data):
        """Convert numpy types and other non-serializable objects to native Python."""
        result = {}
        for key, value in data.items():
            if isinstance(value, (np.integer,)):
                result[key] = int(value)
            elif isinstance(value, (np.floating,)):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    BlackBoxRecorder._make_serializable(v) if isinstance(v, dict)
                    else float(v) if isinstance(v, (np.floating,))
                    else int(v) if isinstance(v, (np.integer,))
                    else v
                    for v in value
                ]
            elif isinstance(value, dict):
                result[key] = BlackBoxRecorder._make_serializable(value)
            elif value == float("inf") or value == float("-inf"):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    @property
    def buffer_frame_count(self):
        """Current number of frames in the buffer."""
        return len(self.frame_buffer)

    @property
    def buffer_duration(self):
        """Current buffer duration in seconds."""
        return len(self.frame_buffer) / max(self.fps, 1)

    def clear(self):
        """Clear the buffer (after saving or for reset)."""
        self.frame_buffer.clear()
        self.data_buffer.clear()
