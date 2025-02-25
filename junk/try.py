from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1,
                        detectionCon=0.5, minTrackCon=0.5)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        for handIndex, handInfo in enumerate(hands):
            lmList = handInfo["lmList"]       # List of 21 landmarks (x,y)
            bbox = handInfo["bbox"]          # (x, y, w, h)
            center = handInfo['center']      # (cx, cy)
            handType = handInfo["type"]      # "Left" or "Right"
            fingers = detector.fingersUp(handInfo)

            # Print how many fingers are up for this hand
            print(f'H{handIndex + 1} = {fingers.count(1)}', end=" ")

            # Loop through each landmark to draw text
            for id, lm in enumerate(lmList):
                x, y = lm[0], lm[1]
                cv2.putText(img, f"{id}",
                            (x, y),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), 2)

        print()  # New line after printing both hands info

    cv2.imshow("Image", img)

    # Press 'ESC' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
