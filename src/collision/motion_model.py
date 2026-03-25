import numpy as np


def estimate_velocity(track, n_last=5):
    pts = track[-n_last:]
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    dt = pts[-1][2] - pts[0][2]

    if dt == 0:
        return 0.0, 0.0

    vx = dx / dt
    vy = dy / dt
    return vx, vy


def predict_future(track, horizon=3.0, step=0.1):
    x, y, _ = track[-1]
    vx, vy = estimate_velocity(track)

    future_positions = []
    times = np.arange(0, horizon, step)

    for dt in times:
        xf = x + vx * dt
        yf = y + vy * dt
        future_positions.append([xf, yf])

    return future_positions, (vx, vy)


class EgoVehicle:
    """
    Ego vehicle model for braking calculations.

    Now uses the full vehicle dynamics simulator (Pacejka tire model,
    ABS, weight transfer) when available. Falls back to v²/2a if
    the dynamics module is not importable.
    """

    def __init__(self, speed=10.0, deceleration=6.0, reaction_time=1.0,
                 surface="dry"):
        """
        Args:
            speed: vehicle speed in m/s
            deceleration: fallback deceleration in m/s² (used only if dynamics module unavailable)
            reaction_time: driver/system reaction time in seconds
            surface: road surface condition ("dry", "wet", "snow", "ice")
        """
        self.speed = speed
        self.deceleration = deceleration
        self.reaction_time = reaction_time
        self.surface = surface
        self._dynamics_available = False

        try:
            from src.vehicle_dynamics.dynamics_sim import braking_distance as _bd
            self._dynamics_braking_distance = _bd
            self._dynamics_available = True
        except ImportError:
            self._dynamics_braking_distance = None

    def braking_distance(self):
        """
        Compute braking distance using the best available model.

        Priority:
        1. Vehicle dynamics simulator (Pacejka + ABS + weight transfer)
        2. Fallback: d = v² / (2a)
        """
        if self._dynamics_available and self._dynamics_braking_distance:
            speed_kmh = self.speed * 3.6  # m/s → km/h
            return self._dynamics_braking_distance(speed_kmh, self.surface)

        # Fallback: simple kinematic model
        return (self.speed ** 2) / (2 * self.deceleration)

    def total_stopping_distance(self):
        """Total distance = reaction distance + braking distance."""
        reaction_distance = self.speed * self.reaction_time
        return reaction_distance + self.braking_distance()
