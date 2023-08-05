from ultralytics import YOLO
import cv2 as cv

model = YOLO('yolov8l.pt')
cap = cv.VideoCapture('videos/bikes.mp4')

while True:
    rec, img = cap.read()
    result = model(img, show=True)
    cv.imshow('frame', img)
    cv.imwrite('Images/object_detection.jpg', result)

cap.release()
cv.destroyAllWindows()