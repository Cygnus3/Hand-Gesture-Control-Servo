import cv2
import mediapipe as mp
import serial
import time


arduino = None
try:
    arduino = serial.Serial('COM3', 9600, timeout=0.1)
    time.sleep(2)  
    print("Arduino Connected")
except:
    print("Arduino not connected. Running in vision-only mode.")



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils



def count_fingers(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

   
    if hand_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)



def calculate_angle(finger_count):
    angle = finger_count * 36
    return max(0, min(180, angle))



def main():
    cap = cv2.VideoCapture(0)
    last_angle = -1

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        finger_count = 0
        angle = 0

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand_label = handedness.classification[0].label  

                
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                
                finger_count = count_fingers(hand_landmarks, hand_label)

                
                angle = calculate_angle(finger_count)

                
                if arduino and angle != last_angle:
                    arduino.write(f"{angle}\n".encode())
                    last_angle = angle

        
        cv2.putText(
            frame,
            f"Hand: {hand_label if results.multi_hand_landmarks else 'None'}",
            (50, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Fingers: {finger_count}  Angle: {angle}",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Hand Gesture Servo Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()