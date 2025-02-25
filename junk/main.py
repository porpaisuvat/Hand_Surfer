import cv2
import time
import numpy as np
import math
import Hand_Track.HandTrackingModule as htm

#########################
wCam, hCam = 640, 360
#########################

# 1. Define reference angle(s) for your pose
#    Example: Suppose we want the angle at the index finger (joint #7) to be around 60 degrees.
REFERENCE_ANGLE = 60.0

# 2. Define a threshold for how close we want to be to the reference angle
#    (The smaller the threshold, the stricter the matching.)
THRESHOLD = 15.0  # degrees

# A helper function to compute the angle between two vectors that share the first point
def compute_angle(p1, p2, p3):
    """
    Compute the angle between vectors (p1 -> p2) and (p1 -> p3) using the dot product formula.
    """
    v1 = (p2[0] - p1[0], p2[1] - p1[1])  # Vector p1 -> p2
    v2 = (p3[0] - p1[0], p3[1] - p1[1])  # Vector p1 -> p3

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # Avoid division by zero
    if mag_v1 < 1e-8 or mag_v2 < 1e-8:
        return 0.0

    cos_theta = dot_product / (mag_v1 * mag_v2)
    # Clip to handle floating-point issues
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle = math.degrees(math.acos(cos_theta))
    return angle

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)     # Width
    cap.set(4, hCam)     # Height

    previous_time = 0

    # 3. Create our hand detector
    detector = htm.handDetector(detectionCon=0.7, maxHands=2)

    while True:
        success, img = cap.read()
        if not success:
            break

        # 4. Detect hands
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # We only proceed if the first hand is detected
        # (lmList[0] should contain landmarks for the first hand)
        if lmList and len(lmList[0]) > 8:
            # We want to measure the angle at landmark #7 using:
            #   joint #7 -> #8 and joint #7 -> #6
            x1, y1 = lmList[0][7][1], lmList[0][7][2]  # (7) base point
            x2, y2 = lmList[0][8][1], lmList[0][8][2]  # (8)
            x3, y3 = lmList[0][6][1], lmList[0][6][2]  # (6)

            # Draw for visualization
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.line(img, (x1, y1), (x3, y3), (255, 0, 255), 2)

            # 5. Compute the current angle
            angle = compute_angle((x1, y1), (x2, y2), (x3, y3))

            # 6. Compute distance to the reference angle
            distance = abs(angle - REFERENCE_ANGLE)

            # 7. Convert distance to a "confidence"
            #    e.g., confidence = 1 - (distance / THRESHOLD), clipped between [0..1]
            confidence = max(0.0, 1.0 - distance/THRESHOLD)

            # 8. Display angle and confidence on the screen
            cv2.putText(img, f'Angle: {int(angle)} deg',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(img, f'Dist to Ref: {distance:.1f}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, f'Confidence: {confidence:.2f}',
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 9. FPS (not crucial for the logic, but for debugging)
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
        previous_time = current_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, hCam-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 10. Show the result
        cv2.imshow("Pose Angle Matching", img)

        # Exit on 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
