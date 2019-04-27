import cv2 as cv

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture('0.MOV')

status, frame = cap.read()

cv.imshow('Frame', frame)


