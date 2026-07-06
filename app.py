import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Initialize Hand Detector (Tracks maximum of 1 hand for drawing)
detector = HandDetector(detectionCon=0.85, maxHands=1)

# Initialize Canvas
canvas = None
prev_x, prev_y = 0, 0

print("Air Canvas Started! Raise your index finger to draw. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Find hands and draw bounding boxes on the frame
    hands, frame = detector.findHands(frame, draw=True)

    if hands:
        hand = hands[0]
        # Get coordinates of the index finger tip (Landmark idx 8)
        lmList = hand["lmList"]
        cx, cy = lmList[8][0], lmList[8][1]

        # Check which fingers are up
        # fingers[1] corresponds to the index finger
        fingers = detector.fingersUp(hand)

        # Draw only if the index finger is UP and middle finger is DOWN (prevents accidental drawing)
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Draw lines on the canvas
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 0), 5)
            prev_x, prev_y = cx, cy
        else:
            # Reset if fingers are not in drawing position
            prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Merge Canvas with live stream
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(threshed)
    
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    frame_fg = cv2.bitwise_and(canvas, canvas, mask=threshed)
    combined_output = cv2.add(frame_bg, frame_fg)

    cv2.imshow("Air Canvas", combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()