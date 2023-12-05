import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize the red color points as a single deque
rpoints = deque(maxlen=1024)

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ignore, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = [[int(lm.x * 640), int(lm.y * 480)] for handslms in result.multi_hand_landmarks for lm in handslms.landmark]
        center = (landmarks[8][0], landmarks[8][1])

        if center[1] <= 65 and 40 <= center[0] <= 140:
            rpoints.clear()
            paintWindow[67:, :] = 255
        elif center[1] > 65:
            rpoints.appendleft(center)

        mpDraw.draw_landmarks(frame, result.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)

    for k in range(1, len(rpoints)):
        cv2.line(frame, rpoints[k - 1], rpoints[k], (255, 0, 0), 20)
        cv2.line(paintWindow, rpoints[k - 1], rpoints[k], (255, 0, 0), 20)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()