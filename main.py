#coding:UTF-8

import cv2

import mediapipe as mp


cap = cv2.VideoCapture(0)

mpHolistic = mp.solutions.holistic

holistic = mpHolistic.Holistic()

mpDraw = mp.solutions.drawing_utils

drawing_spec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

while True :

    success, image = cap.read()

    if not success:

        break

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    RESULTS = holistic.process(imageRGB)

    mpDraw.draw_landmarks(image, RESULTS.face_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)

    mpDraw.draw_landmarks(image, RESULTS.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)

    mpDraw.draw_landmarks(image, RESULTS.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)

    mpDraw.draw_landmarks(image, RESULTS.pose_landmarks, mpHolistic.POSE_CONNECTIONS, drawing_spec, drawing_spec)

    cv2.imshow('MASTER LIPAKUMU BODY DETECTOR', image)

    if cv2.waitKey(1) & 0xff == ord('q'):

        break
#MasterLipakumu 