import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, 1, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        handTypes = []  # Store "Left" or "Right"

        if self.results.multi_hand_landmarks:
            for idx, handLms in enumerate(self.results.multi_hand_landmarks):
                label = self.results.multi_handedness[idx].classification[0].label
                handTypes.append(label)  # "Right" or "Left"

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img, handTypes  # Return image and hand labels

    def findPosition(self, img, draw=True):
        lmLists = []  # List to store landmarks for all detected hands

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])  # Append each landmark
                lmLists.append(lmList)  # Append full hand landmarks

        return lmLists  # Returns a list of landmark lists (one per hand)
