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
    def __init__(self, speed=10.0, deceleration=6.0, reaction_time=1.0):
        self.speed = speed
        self.deceleration = deceleration
        self.reaction_time = reaction_time

    def braking_distance(self):
        return (self.speed ** 2) / (2 * self.deceleration)

    def total_stopping_distance(self):
        reaction_distance = self.speed * self.reaction_time
        return reaction_distance + self.braking_distance()
