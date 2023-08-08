from ultralytics import YOLO
import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture('videos/cars.mp4')
model = YOLO('yolov8n.pt')

mask = cv.imread('Images/mask_carcounter.png')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)  # combine video with mask
    results = model(imgRegion, stream=True)  # apply model to image

    detections = np.empty((0, 5))  # create an empty array

    for result in results:
        boxes = result.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]  # to find the desired object name

            # Check if the current class is car, motorbike, tractor, or bus
            if currentClass in ['car', 'motorbike', 'tractor', 'bus'] and conf > 0.2:
                label = f"{currentClass}: {conf}"
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv.putText(img, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv.imshow('Image', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()