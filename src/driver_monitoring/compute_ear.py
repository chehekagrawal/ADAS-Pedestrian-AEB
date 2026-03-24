import math

def euclidean_distance(ptA, ptB):
    return math.sqrt((ptA[0] - ptB[0])**2 + (ptA[1] - ptB[1])**2)

def compute_ear(eye_points):
    """
    Computes the Eye Aspect Ratio (EAR) given 6 face landmark points.
    Based on Soukupová and Čech 2016.
    
    Parameters:
    eye_points: List or array of 6 (x, y) coordinates.
                Points are ordered clockwise starting from the left corner
                of the given eye.
                p1 = eye_points[0]
                p2 = eye_points[1]
                p3 = eye_points[2]
                p4 = eye_points[3]
                p5 = eye_points[4]
                p6 = eye_points[5]
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = euclidean_distance(eye_points[1], eye_points[5])  # p2-p6
    B = euclidean_distance(eye_points[2], eye_points[4])  # p3-p5

    # Compute the euclidean distance between the horizontal eye landmark (p1-p4)
    C = euclidean_distance(eye_points[0], eye_points[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear
