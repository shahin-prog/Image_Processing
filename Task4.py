from ultralytics import YOLO
import cv2 as cv
import numpy as np
import math
from sort import *

cap = cv.VideoCapture('videos/cars.mp4')
model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
             "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "motorcycle", "tractor", "bus"]

mask = cv.imread('Images/mask_carcounter.png')

# Tracking
tracker = Sort(max_age=20)

limits = [200, 397, 673, 397]  # define a line to count the number of cars
totalCount = {'car': [], 'motorbike': [], 'tractor': [], 'bus': []}

while True:
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)  # combine video with mask
    results = model(imgRegion, stream=True)  # apply model to image

    detections = np.empty((0, 5))  # create an empty array


    for result in results:
        boxes = result.boxes
        for box in boxes:
            print(box)

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]  # to find the desired object name

            if currentClass in totalCount and conf > 0.3:
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # line
    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(result)
        w, h = x2 - x1, y2 - y1

        cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        image = cv.putText(img, f'{id}', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        centerX, centerY = x1 + w // 2, y1 + h // 2  # to find object's center
        cv.circle(img, (centerX, centerY), 5, (255, 0, 255), cv.FILLED)

        if limits[0] < centerX < limits[2] and limits[1] - 15 < centerY < limits[1] + 15:
            if totalCount[currentClass].count(id) == 0:
                totalCount[currentClass].append(id)
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv.putText(img, f'Car: {len(totalCount["car"])}', (255, 100), cv.FONT_HERSHEY_SIMPLEX, 5, (50, 50, 255), 8)
    cv.putText(img, f'Motorbike: {len(totalCount["motorbike"])}', (255, 200), cv.FONT_HERSHEY_SIMPLEX, 5, (50, 50, 255), 8)
    cv.putText(img, f'Tractor: {len(totalCount["tractor"])}', (255, 300), cv.FONT_HERSHEY_SIMPLEX, 5, (50, 50, 255), 8)
    cv.putText(img, f'Bus: {len(totalCount["bus"])}', (255, 400), cv.FONT_HERSHEY_SIMPLEX, 5, (50, 50, 255), 8)

    cv.imshow('Image', img)
    cv.waitKey(1)