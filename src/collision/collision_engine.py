import numpy as np
from shapely.geometry import Point, Polygon


def create_vehicle_box(center_x, center_y, length=4.5, width=2.0):
    return Polygon([
        (center_x - length / 2, center_y - width / 2),
        (center_x + length / 2, center_y - width / 2),
        (center_x + length / 2, center_y + width / 2),
        (center_x - length / 2, center_y + width / 2),
    ])


def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_relative_speed(vehicle_speed, pedestrian_velocity):
    ped_speed = np.linalg.norm(np.array(pedestrian_velocity))
    return vehicle_speed + ped_speed


def compute_ttc(distance, relative_speed):
    if relative_speed <= 0:
        return float("inf")
    return distance / relative_speed


def check_collision(vehicle_poly, ped_pos, radius=0.4):
    pedestrian = Point(ped_pos[0], ped_pos[1]).buffer(radius)
    return vehicle_poly.intersects(pedestrian)


def collision_probability(ttc):
    if ttc < 1:
        return 0.9
    elif ttc < 2:
        return 0.6
    elif ttc < 3:
        return 0.3
    else:
        return 0.05
