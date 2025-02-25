import cv2
import numpy as np
import HandTrackingModule as htm
from Math.angle_math import compute_angle
from Math.csv_utils import get_file_name, write_csv

# -----------------------------------------------------
# 1) Define Joint Triplets (3x5 Matrix)
# -----------------------------------------------------
JOINTS_MATRIX = [
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)],  # Thumb
    [(0, 5, 6), (5, 6, 7), (6, 7, 8)],  # Index
    [(0, 9, 10), (9, 10, 11), (10, 11, 12)],  # Middle
    [(0, 13, 14), (13, 14, 15), (14, 15, 16)],  # Ring
    [(0, 17, 18), (17, 18, 19), (18, 19, 20)]  # Pinky
]

# -----------------------------------------------------
# 2) Compute Spread & Wrist Position
# -----------------------------------------------------
def compute_spread(pinky, index, thumb):
    """Calculate the spread between fingers."""
    spread_index_pinky = np.linalg.norm(np.array(index) - np.array(pinky))
    spread_thumb_pinky = np.linalg.norm(np.array(thumb) - np.array(pinky))
    return spread_index_pinky, spread_thumb_pinky

def compute_wrist_position(wrist):
    """Return wrist position (Y-coordinate as height reference)."""
    return wrist[1]  # Only return Y-position

# -----------------------------------------------------
# 3) Initialize Camera & Hand Detector
# -----------------------------------------------------
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.7, maxHands=2)  # ✅ Detect both hands

# -----------------------------------------------------
# 4) Prepare CSV Header (Left & Right Hand Data in Same Row)
# -----------------------------------------------------
hand_features = [f"A{i}{j}" for i in range(3) for j in range(5)] + ["Spread_IndexPinky", "Spread_ThumbPinky", "Wrist_Position"]
header = ["Frame"] + [f"L_{feature}" for feature in hand_features] + [f"R_{feature}" for feature in hand_features]

file_name = get_file_name()
data = []

frame_count = 0  # ✅ Frame counter
max_frames = 1000  # ✅ Stop after 1000 frames

print("\n📸 Show both hands in the camera...")
print(f"🛑 Data collection will stop after {max_frames} frames or if you press 'ESC'.")

while frame_count < max_frames:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # ✅ Fix mirrored camera issue

    # 1) Detect Hands & Get Hand Type (Right/Left)
    img, handTypes = detector.findHands(img)
    lmLists = detector.findPosition(img)  # ✅ Now returns landmarks for all hands

    # Initialize Empty Data for Left & Right Hands
    left_hand_data = [np.nan] * (len(hand_features))  # Missing values if left hand not detected
    right_hand_data = [np.nan] * (len(hand_features))  # Missing values if right hand not detected

    # 2) Process Each Hand
    if lmLists:
        for i, lmList in enumerate(lmLists):  # ✅ Loop through detected hands
            if len(lmList) < 21:
                continue  # Skip if not enough landmarks

            hand_label = handTypes[i] if isinstance(handTypes, list) else handTypes  # "Right" or "Left"

            # 3) Compute Angles (3x5 Matrix)
            angle_matrix = []
            for finger in JOINTS_MATRIX:
                row = []
                for (start, mid, end) in finger:
                    x_mid, y_mid = lmList[mid][1], lmList[mid][2]
                    x_start, y_start = lmList[start][1], lmList[start][2]
                    x_end, y_end = lmList[end][1], lmList[end][2]

                    angle = compute_angle((x_mid, y_mid), (x_start, y_start), (x_end, y_end))
                    row.append(angle)
                angle_matrix.append(row)

            # 4) Compute Spread & Wrist Position
            pinky = lmList[20][1:3]  # Pinky tip
            index = lmList[8][1:3]  # Index tip
            thumb = lmList[4][1:3]  # Thumb tip
            wrist = lmList[0][1:3]  # Wrist position

            spread_index_pinky, spread_thumb_pinky = compute_spread(pinky, index, thumb)
            wrist_position = compute_wrist_position(wrist)

            # 5) Flatten Angles & Add Extra Features
            flat_angles = [angle for row in angle_matrix for angle in row]
            hand_data = flat_angles + [spread_index_pinky, spread_thumb_pinky, wrist_position]

            # 6) Store in Correct Hand Column
            if hand_label == "Left":
                left_hand_data = hand_data
            elif hand_label == "Right":
                right_hand_data = hand_data

    # 7) Store Data in a Single Row (Frame, Left Hand Data, Right Hand Data)
    both_hands_present = not (np.isnan(left_hand_data[0]) or np.isnan(right_hand_data[0]))

    if both_hands_present:
        data.append([frame_count] + left_hand_data + right_hand_data)
        frame_count += 1  # ✅ Increase frame count
        print(f"👐 Both hands detected! Frame {frame_count} saved.")  # Debugging info
    else:
        print("❌ Only one hand detected, frame skipped.")  # Optional debug message

    # 8) Show Live Video Feed with Frame Count
    cv2.putText(img, f"Frames Collected: {frame_count}/{max_frames}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow("Collecting Hand Pose Data", img)

    # ✅ Exit if 'ESC' is pressed or 1000 frames collected
    if cv2.waitKey(1) & 0xFF == 27 or frame_count >= max_frames:
        break

# 9) Save Data to CSV
write_csv(file_name, header, data)

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Data collection stopped at {frame_count} frames. Data saved in '{file_name}'")
