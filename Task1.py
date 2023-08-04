import cv2 as cv

img = cv.imread('Images/dog.jpg', cv.IMREAD_COLOR)

text = 'Dog'
org = (20, 60)
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 0, 255)
thickness = 2

cv.putText(img, text, org, font, fontScale, color, thickness)
cv.rectangle(img, (25,70), (140,230),(255,255,255), 3)
cv.imshow('Image Show', img)
cv.imwrite('Image/ax_1.jpg', img)

cv.waitKey(0)
cv.destroyAllWindows()