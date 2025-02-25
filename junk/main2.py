import cv2
import time
import math
import csv
import numpy as np
import Hand_Track.HandTrackingModule as htm  # Ensure HandTrackingModule can return handedness

# -----------------------------------------------------
# 1) Define Joint Triplets in a 3x5 Structure (3 angles per finger)
# -----------------------------------------------------
JOINTS_MATRIX = [
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)],  # Thumb
    [(0, 5, 6), (5, 6, 7), (6, 7, 8)],  # Index
    [(0, 9, 10), (9, 10, 11), (10, 11, 12)],  # Middle
    [(0, 13, 14), (13, 14, 15), (14, 15, 16)],  # Ring
    [(0, 17, 18), (17, 18, 19), (18, 19, 20)]  # Pinky
]

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

# -----------------------------------------------------
# 2) Preprocessing Functions for ANN
# -----------------------------------------------------
def normalize_angles(angle_list):
    """Normalize angles to [0,1] for ANN."""
    return [(a / 180.0) for a in angle_list]  # 180° is the max possible

def preprocess_angles(angle_list):
    """Apply all preprocessing steps."""
    return normalize_angles(angle_list)

def main():
    # -----------------------------------------------------
    # 3) Initialize Camera & Hand Detector
    # -----------------------------------------------------
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(detectionCon=0.7, maxHands=1)  # Set to detect only 1 hand

    # -----------------------------------------------------
    # 4) Create and Open the CSV File
    # -----------------------------------------------------
    with open("right_hand_angles.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # (a) Create header row (3x5 matrix = 15 columns)
        header = [f"A{i}{j}" for i in range(3) for j in range(5)]
        writer.writerow(header)

        while True:
            success, img = cap.read()
            if not success:
                break

            # 1) Detect Hands and Get Handedness
            img = detector.findHands(img)
            lmList, handType = detector.findPosition(img, draw=False, returnHandedness=True)  

            # 2) Check if a right hand is detected
            if lmList and len(lmList[0]) >= 21 and handType == "Right":
                firstHand = lmList[0]

                # 3) Compute angles for each joint in 3x5 matrix form
                angle_matrix = []
                for finger in JOINTS_MATRIX:
                    row = []
                    for (start, mid, end) in finger:
                        x_mid, y_mid = firstHand[mid][1], firstHand[mid][2]
                        x_start, y_start = firstHand[start][1], firstHand[start][2]
                        x_end, y_end = firstHand[end][1], firstHand[end][2]

                        angle = compute_angle(
                            (x_mid, y_mid),
                            (x_start, y_start),
                            (x_end, y_end)
                        )
                        row.append(angle)
                    angle_matrix.append(row)

                # 4) Flatten Matrix to Single Row for CSV
                flat_angles = [angle for row in angle_matrix for angle in row]

                # 5) Apply Preprocessing
                flat_angles = preprocess_angles(flat_angles)

                # 6) Write row to CSV
                writer.writerow(flat_angles)

            # 7) Show Live Video Feed
            cv2.imshow("Right Hand Angles", img)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
