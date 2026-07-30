from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import cv2
from deepface import DeepFace
import tempfile

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/verify")
async def verify(profile_img: UploadFile, selfie_video: UploadFile):
    # שומר זמני
    with tempfile.NamedTemporaryFile() as p, tempfile.NamedTemporaryFile() as s:
        p.write(await profile_img.read()); s.write(await selfie_video.read())

        # מוציא פריים מהסלפי
        cap = cv2.VideoCapture(s.name); ret, frame = cap.read(); cap.release()
        cv2.imwrite("temp_frame.jpg", frame)

        # 1. האם זה אותו בן אדם
        match = face_recognition.compare_faces(
            [face_recognition.face_encodings(face_recognition.load_image_file(p.name))[0]],
            face_recognition.face_encodings(cv2.imread("temp_frame.jpg"))[0]
        )[0]

        # 2. גיל + מין מהסלפי
        analysis = DeepFace.analyze(img_path = "temp_frame.jpg", actions = ['age', 'gender'], enforce_detection=False)[0]
        real_age = analysis['age']
        real_gender = analysis['gender']

    return {
        "is_same_person": match,
        "detected_age": real_age,
        "detected_gender": real_gender,
        "approved": match # אם לא אותו בן אדם = נדחה אוטומטי
    }
