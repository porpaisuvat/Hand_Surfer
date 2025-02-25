import cv2
import time
import numpy as np
import Hand_Track.HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
wCam, hCam = 640, 480
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # 1) Detect hand & get landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # 2) Check we have enough landmarks
    if len(lmList) >= 9:
        # Thumb (id=4)
        x1, y1 = lmList[4][1], lmList[4][2]
        # Index finger (id=8)
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw circles/line
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Distance between thumb and index
        length = math.hypot(x2 - x1, y2 - y1)

        # 3) Convert the distance to volume range
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        print(f"Distance={int(length)}, Volume={vol}")
        volume.SetMasterVolumeLevel(vol, None)

        # Visual feedback when fingers are close
        if length < 10:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # 4) Volume Bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # 5) FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)

    # Press ESC to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
