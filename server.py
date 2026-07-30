from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import cv2
import numpy as np
import tempfile

app = FastAPI()

# מאפשר לאפליקציה שלך לשלוח בקשות לשרת
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify(profile_img: UploadFile = File(...), selfie_video: UploadFile = File(...)):
    try:
        # שמירת קבצים זמנית
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as p:
            p.write(await profile_img.read())
            profile_path = p.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as s:
            s.write(await selfie_video.read())
            selfie_path = s.name

        # מוציא פריים ראשון מהווידאו
        cap = cv2.VideoCapture(selfie_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"error": "לא הצלחתי לקרוא את הסלפי"}

        # טוען תמונות
        profile_image = face_recognition.load_image_file(profile_path)
        selfie_image = frame

        # מוצא קידוד פנים
        profile_enc = face_recognition.face_encodings(profile_image)
        selfie_enc = face_recognition.face_encodings(selfie_image)

        if len(profile_enc) == 0:
            return {"error": "לא נמצאו פנים בתמונת פרופיל"}

        if len(selfie_enc) == 0:
            return {"error": "לא נמצאו פנים בסלפי"}

        # משווה בין הפנים
        distance = face_recognition.face_distance([profile_enc[0]], selfie_enc[0])[0]
        confidence = round((1 - distance) * 100, 2)
        is_same = confidence > 75

        return {
            "is_same_person": is_same,
            "confidence": confidence,
            "approved": is_same
        }

    except Exception as e:
        return {"error": str(e)}
