
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import os, subprocess, json, random
import tensorflow as tf
import numpy as np
import editdistance
from time import time
from utils import load_video, load_alignments, num_to_char, load_data
from modelutil import load_model
# from pydantic import BaseModel

# Set seeds and force determinism
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

save_path = "accuracy_results.json"
model = load_model()
model.load_weights("checkpoints.weights.h5")

DATA_DIR = "data/s1"

@app.get("/videos/")
def list_mpg_videos():
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".mpg")]




@app.get("/predict/")
def predict_lip(video_name: str = Query(...)):
    file_base = os.path.splitext(video_name)[0]
    mpg_path = os.path.join(DATA_DIR, video_name)
    mp4_path = os.path.join("temp", f"{file_base}.mp4")
    os.makedirs("temp", exist_ok=True)

    subprocess.run(["ffmpeg", "-y", "-i", mpg_path, mp4_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frames, alignments = load_data(tf.convert_to_tensor(mpg_path))

    real_text = tf.strings.reduce_join([num_to_char(char) for char in alignments]).numpy().decode('utf-8')

    yhat = model.predict(tf.expand_dims(frames, axis=0), verbose=0)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

    predicted_text = tf.strings.reduce_join([num_to_char(char) for char in decoded[0]]).numpy().decode('utf-8')

    # Add timestamp query param to prevent caching
    timestamp = int(time())
    video_url = f"http://localhost:8000/temp/{file_base}.mp4?ts={timestamp}"

    return {
        "real_text": real_text.strip(),
        "predicted_text": predicted_text.strip(),
        "video_url": video_url
    }


app.mount("/temp", StaticFiles(directory="temp"), name="temp")

accuracy_file = "accuracy_cache.json"

@app.get("/calculate-accuracy")
def read_cached_accuracy():
    if os.path.exists(accuracy_file):
        with open(accuracy_file, "r") as f:
            return json.load(f)
    return {
        "error": "Accuracy cache not found. Please generate accuracy_cache.json manually."
    }

