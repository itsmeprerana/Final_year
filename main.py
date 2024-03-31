from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import numpy as np
from typing import List
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import librosa
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMOTION_MAPPING = {
        'YAF_angry': 'ANGRY',
        'YAF_disgust': 'DISGUST',
        'YAF_fear': 'FEAR',
        'YAF_happy': 'HAPPY',
        'YAF_neutral': 'NEUTRAL',
        'YAF_pleasant_surprised': 'SURPRISED',
        'YAF_sad': 'SAD',
        'OAF_angry': 'ANGRY',
        'OAF_disgust': 'DISGUST',
        'OAF_Fear': 'FEAR',
        'OAF_happy': 'HAPPY',
        'OAF_neutral': 'NEUTRAL',
        'OAF_Pleasant_surprised': 'SURPRISED',
        'OAF_Sad': 'SAD',
    }

model = None
label_encoder = None

def load_model_and_label_encoder():
    global model, label_encoder
    # Load the trained model and label encoder
    model = load_model("prerana.h5")  # Replace with the actual path to your model file
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("prerana.npy")  # Replace with the actual path to your label encoder file

# Load the model and label encoder on startup
load_model_and_label_encoder()

def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Save the uploaded file
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    # Process the uploaded file
    try:
        features = extract_features(file.filename)
        features = features[np.newaxis, np.newaxis, :]

        # Make prediction using the loaded model
        predicted_probabilities = model.predict(features)
        predicted_label_index = np.argmax(predicted_probabilities)
        predicted_emotion = label_encoder.classes_[predicted_label_index]

        return JSONResponse(content={"filename": file.filename, "predicted_emotion":  EMOTION_MAPPING.get(predicted_emotion)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
