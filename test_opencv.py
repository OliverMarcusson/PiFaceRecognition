import cv2 

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgbackground = cv2.imread('resources/UI face checking.jpg')

while True:
    success, img = cap.read()
    
    imgbackground[162:162+480, 55:55+640] = img 
    
    cv2.imshow("Face recognition", imgbackground)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)