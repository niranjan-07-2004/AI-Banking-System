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
# SIMULATED BANK DATA
# ===============================
balance = 5000

# ===============================
# INITIALIZE FIREBASE
# ===============================
cred = credentials.Certificate("ai-banking-system-firebase-adminsdk-fbsvc-bb5c32abf3.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ===============================
# CLOUD LOG FUNCTION
# ===============================
def log_event(user, status):
    data = {
        "user": user,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    db.collection("login_logs").add(data)

# ===============================
# BANKING MENU
# ===============================
def banking_menu():
    global balance

    while True:
        print("\n🏦 Secure Banking Menu")
        print("1. Check Balance")
        print("2. Withdraw")
        print("3. Deposit")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            print("💰 Current Balance:", balance)
            log_event(AUTHORIZED_USER, f"Checked Balance: {balance}")

        elif choice == "2":
            amt = int(input("Enter amount to withdraw: "))
            if amt <= balance:
                balance -= amt
                print("✅ Withdrawal Successful")
                print("💰 Remaining Balance:", balance)
                log_event(AUTHORIZED_USER, f"Withdrawn: {amt}")
            else:
                print("❌ Insufficient Balance")

        elif choice == "3":
            amt = int(input("Enter amount to deposit: "))
            balance += amt
            print("✅ Deposit Successful")
            print("💰 Updated Balance:", balance)
            log_event(AUTHORIZED_USER, f"Deposited: {amt}")

        elif choice == "4":
            print("👋 Exiting Banking Menu")
            break

        else:
            print("❌ Invalid choice")

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
print("Press 'B' for Banking Menu after login")
print("Press 'Q' to quit")

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
        # CHECK FRAUD LOCK
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

            cv2.putText(frame, "Press B for Banking Menu", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

    key = cv2.waitKey(1) & 0xFF

    if key == ord('b') and not locked:
        banking_menu()

    if key == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cam.release()
cv2.destroyAllWindows()
print("System closed safely.")
