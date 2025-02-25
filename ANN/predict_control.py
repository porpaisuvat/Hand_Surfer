import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time
import sys
import os
import datetime
import win32gui
import win32con
import win32api
from pynput.keyboard import Key, Controller

sys.path.append(os.path.abspath("../Hand_Track/"))
import HandTrackingModule as htm
sys.path.append(os.path.abspath("../Hand_Track/Math/")) 
from Math.angle_math import compute_angle

# -----------------------------------------------------
# 1) Load Latest Model & Preprocessing Objects
# -----------------------------------------------------
models_dir = "models"
latest_folder = max([os.path.join(models_dir, d) for d in os.listdir(models_dir)], key=os.path.getmtime)

model_path = os.path.join(latest_folder, "model.h5")
label_encoder_path = os.path.join(latest_folder, "label_encoder.npy")
scaler_mean_path = os.path.join(latest_folder, "scaler_mean.npy")
scaler_scale_path = os.path.join(latest_folder, "scaler_scale.npy")

model = tf.keras.models.load_model(model_path)
label_encoder = np.load(label_encoder_path, allow_pickle=True)

scaler_mean = np.load(scaler_mean_path, allow_pickle=True)
scaler_scale = np.load(scaler_scale_path, allow_pickle=True)

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale  # Fix: Add scale_

print(f"\n✅ Loaded model from: {latest_folder}")

# -----------------------------------------------------
# 2) Define Constants & Hand Tracking
# -----------------------------------------------------
JOINTS_MATRIX = [
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)],  # Thumb
    [(0, 5, 6), (5, 6, 7), (6, 7, 8)],  # Index
    [(0, 9, 10), (9, 10, 11), (10, 11, 12)],  # Middle
    [(0, 13, 14), (13, 14, 15), (14, 15, 16)],  # Ring
    [(0, 17, 18), (17, 18, 19), (18, 19, 20)]  # Pinky
]

keyboard = Controller()  # ✅ Initialize Keyboard Controller
last_move = "standby"  # ✅ Track last executed move

def compute_spread(pinky, index, thumb):
    """Calculate the spread between fingers."""
    spread_index_pinky = np.linalg.norm(np.array(index) - np.array(pinky))
    spread_thumb_pinky = np.linalg.norm(np.array(thumb) - np.array(pinky))
    return spread_index_pinky, spread_thumb_pinky

def compute_wrist_position(wrist):
    """Return wrist position (Y-coordinate as height reference)."""
    return wrist[1] 
# -----------------------------------------------------
# 3) Define Key Press Functions for BlueStacks
# -----------------------------------------------------
def bring_bluestacks_to_front():
    hwnd = win32gui.FindWindow(None, "BlueStacks App Player")
    if hwnd:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        
    else:
        print("Could not find the BlueStacks window")


def left():
    bring_bluestacks_to_front()
    keyboard.press(Key.left)
    keyboard.release(Key.left)
    print("⬅️ Moving Left")

def right():
    bring_bluestacks_to_front()
    keyboard.press(Key.right)
    keyboard.release(Key.right)
    print("➡️ Moving Right")

def jump():
    bring_bluestacks_to_front()
    keyboard.press(Key.up)
    keyboard.release(Key.up)
    print("🆙 Jumping")

def roll():
    bring_bluestacks_to_front()
    keyboard.press(Key.down)
    keyboard.release(Key.down)
    print("🔄 Rolling")

def space():
    bring_bluestacks_to_front()
    keyboard.press(Key.space)
    keyboard.release(Key.space)
    time.sleep(0.100)
    keyboard.press(Key.space)
    keyboard.release(Key.space)
    print("🛹 Hoverboard Activated")

# -----------------------------------------------------
# 4) Initialize Camera & Hand Detector
# -----------------------------------------------------
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.9, maxHands=2)  # ✅ Detect both hands

print("\n🖐️ Waiting for both hands to appear...")

# Wait until both hands are detected
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # Fix mirrored camera
    img, handTypes = detector.findHands(img)
    lmLists = detector.findPosition(img)

    # Ensure both hands are present before starting
    if lmLists and len(lmLists) == 2:  # Ensure exactly 2 hands are detected
        print("\n✅ Both hands detected! Starting prediction...")
        time.sleep(1)  # Short delay before starting predictions
        break

# -----------------------------------------------------
# 5) Real-Time Gesture Prediction & Game Control
# -----------------------------------------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Fix mirrored camera issue
    img, handTypes = detector.findHands(img)
    lmLists = detector.findPosition(img)

    # Ensure both hands are present
    if lmLists and len(lmLists) == 2:
        left_hand_data = [np.nan] * (len(JOINTS_MATRIX) * 3 + 3)  # Empty placeholders
        right_hand_data = [np.nan] * (len(JOINTS_MATRIX) * 3 + 3)  # Empty placeholders

        for i, lmList in enumerate(lmLists):
            if len(lmList) < 21:
                continue  # Skip if not enough landmarks

            hand_label = handTypes[i]  # "Right" or "Left"

            # Compute angles
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

            # Compute spread and wrist position
            pinky = lmList[20][1:3]  # Pinky tip
            index = lmList[8][1:3]  # Index tip
            thumb = lmList[4][1:3]  # Thumb tip
            wrist = lmList[0][1:3]  # Wrist position

            spread_index_pinky, spread_thumb_pinky = compute_spread(pinky, index, thumb)
            wrist_position = compute_wrist_position(wrist)

            # Flatten data
            flat_angles = [angle for row in angle_matrix for angle in row]
            hand_data = flat_angles + [spread_index_pinky, spread_thumb_pinky, wrist_position]

            # Assign data to correct hand slot
            if hand_label == "Left":
                left_hand_data = hand_data
            elif hand_label == "Right":
                right_hand_data = hand_data

        # Combine both hands' data
        input_data = np.array([left_hand_data + right_hand_data])  # Shape (1, num_features)

        # Normalize the input data
        input_data = scaler.transform(input_data)

        # Predict using the model
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = label_encoder[predicted_class]

        if confidence < 0.7 or predicted_label == "standby":
            print("🟡 Standby (Low confidence)")
            last_move = "standby"
        elif last_move == "standby":
            print(f"🟢 Prediction: {predicted_label} ({confidence:.2f})")
            cv2.putText(img, predicted_label, (40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
            if predicted_label == "left_fist_hook":
                left()
            elif predicted_label == "right_fist_hook":
                right()
            elif predicted_label == "fingers_bend":
                roll()
            elif predicted_label == "finger_spread":
                jump()
            elif predicted_label == "ermm":
                space()
            last_move = predicted_label

    cv2.imshow("Real-Time Gesture Prediction", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n🛑 Prediction stopped.")
