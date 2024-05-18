import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

cap = cv2.VideoCapture(0)
#run
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        if len(results.multi_handedness) == 2:
            cv2.putText(img, 'Both Hands', (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        else:
            for handedness in results.multi_handedness:
                label_dict = MessageToDict(handedness)
                hand_label = label_dict['classification'][0]['label']
                x_position = 20 if hand_label == 'Left' else 460
                cv2.putText(img, hand_label + ' Hand', (x_position, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

