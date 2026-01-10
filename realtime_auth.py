import cv2
import joblib
import numpy as np
import time

# ===============================
# FIREBASE IMPORTS
# ===============================
import firebase_admin
from firebase_admin import credentials, firestore

# ===============================
# CONFIGURATION
# ===============================
AUTHORIZED_USER = "Gopal"     # change if needed
MAX_ATTEMPTS = 3              # fraud threshold
LOCK_TIME = 10                # seconds

# ===============================
# INITIALIZE FIREBASE
# ===============================
cred = credentials.Certificate("ai-banking-system-firebase-adminsdk-fbsvc-bb5c32abf3.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ===============================
# LOG FUNCTION (CLOUD DATABASE)
# ===============================
def log_event(user, status):
    data = {
        "user": user,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    db.collection("login_logs").add(data)

# ===============================
# LOAD ML MODELS
# ===============================
svm = joblib.load("svm_face_model.pkl")
knn = joblib.load("knn_face_model.pkl")

# ===============================
# LOAD FACE DETECTOR
# ===============================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# FRAUD VARIABLES
# ===============================
failed_attempts = 0
locked = False
lock_start_time = None

# ===============================
# START WEBCAM
# ===============================
cam = cv2.VideoCapture(0)

print("📷 AI Banking System Started")
print("Press 'q' to quit")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50))
        face = face.flatten().reshape(1, -1)

        # ===============================
        # CHECK IF SYSTEM IS LOCKED
        # ===============================
        if locked:
            if current_time - lock_start_time >= LOCK_TIME:
                locked = False
                failed_attempts = 0
            else:
                label = "FRAUD ALERT - SYSTEM LOCKED"
                color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                continue

        # ===============================
        # MODEL PREDICTION
        # ===============================
        svm_pred = svm.predict(face)[0]
        knn_pred = knn.predict(face)[0]

        # ===============================
        # AUTHENTICATION LOGIC
        # ===============================
        if svm_pred == knn_pred == AUTHORIZED_USER:
            label = "ACCESS GRANTED"
            color = (0, 255, 0)
            failed_attempts = 0

            log_event(AUTHORIZED_USER, "ACCESS GRANTED")

        else:
            failed_attempts += 1
            label = f"ACCESS DENIED ({failed_attempts})"
            color = (0, 0, 255)

            log_event("Unknown", "ACCESS DENIED")

            if failed_attempts >= MAX_ATTEMPTS:
                locked = True
                lock_start_time = current_time
                label = "FRAUD ALERT - SYSTEM LOCKED"

                log_event("Unknown", "FRAUD ALERT")

        # ===============================
        # DISPLAY RESULT
        # ===============================
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("AI Banking System - Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cam.release()
cv2.destroyAllWindows()
print("System closed safely.")
