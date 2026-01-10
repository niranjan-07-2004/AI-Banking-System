import cv2
import os

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

user_id = input("Enter User ID: ")
path = f"face_data/{user_id}"

os.makedirs(path, exist_ok=True)

count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite(f"{path}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Capturing Face", frame)

    if cv2.waitKey(1) == 27 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
