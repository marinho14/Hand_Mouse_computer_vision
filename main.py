import cv2
import mediapipe as mp
import mouse
import ctypes
import pandas
import time
import datetime

import pandas as pd

a=0
Tiempo=0
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
ancho, alto = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0) ##The number correspond to the port
with mp_hands.Hands(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as hands:
  while cap.isOpened():
    success, image = cap.read() #It will read image as BGR
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    #cv2.imshow('Video', image)

    # Flip the image horizontally for a later selfie-view display, and convert
    # Conversion from BGR image to RGB (C).
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image) #Get the landmarks of hand

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(image,hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
        X = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * ancho)
        Y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * alto)

        #Movimiento para dar click
        Y_1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
        Y_2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)

        X_1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_height)
        X_2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_height)
        Tiempo= Tiempo+1
        X_T= X_1-X_2
        Y_T=Y_1-Y_2
        print("X",Y_T)
        print("Y",X_T)
        if abs(X_T)<30 and abs(Y_T)< 30 and Tiempo==5:
          mouse.click("left")
          print("CLICK",a)
          Tiempo=0


        X_mouse= int(X*ancho)
        Y_mouse= int(Y*alto)
        cv2.circle(image, (X, Y),3, color=(0, 0, 255), thickness=2)
        mouse.move(X,Y)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()