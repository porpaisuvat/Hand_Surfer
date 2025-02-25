import math
import numpy as np

def compute_angle(p_mid, p_start, p_end):
    """Compute the angle (in degrees) at p_mid formed by vectors:
       p_mid->p_start and p_mid->p_end using the dot product."""
    v1 = (p_start[0] - p_mid[0], p_start[1] - p_mid[1])
    v2 = (p_end[0] - p_mid[0], p_end[1] - p_mid[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 < 1e-8 or mag2 < 1e-8:
        return 0.0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Avoid floating-point error
    return math.degrees(math.acos(cos_angle))

def normalize_angles(angle_list):
    """Normalize angles to [0,1] for ANN."""
    return [(a / 180.0) for a in angle_list]

def preprocess_angles(angle_list):
    """Apply preprocessing: normalize & remove outliers."""
    return normalize_angles(angle_list)
