# ---------- Single Audio File Evaluation----------
# !pip install -q librosa soundfile

import os
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DRIVE_WORKDIR = "/content/drive/MyDrive/major/project/dataset"
OUT_MFCC_DIR = os.path.join(DRIVE_WORKDIR, "data/mfcc_augmented")
SAVE_MODEL_PATH = os.path.join(DRIVE_WORKDIR, "best_model.keras")
LAST_CHECKPOINT = os.path.join(DRIVE_WORKDIR, "last_model.keras")

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LEN = 862
IMG_SIZE = (128, 128)
CLASS_FOLDER = os.path.join(OUT_MFCC_DIR, "train")

# Path to the audio file you want to test
AUDIO_PATH = "/content/drive/MyDrive/major/project/dataset/ICBHI_final_database/159_1b1_Ar_sc_Meditron.wav"

# ---------- Helper Functions ----------
def discover_classes(train_folder):
    """Get class names from train/<class> folders"""
    p = Path(train_folder)
    return sorted([d.name for d in p.iterdir() if d.is_dir()])

def compute_mfcc_for_file(path, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """Compute MFCC and pad/truncate"""
    y, sr_loaded = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.astype(np.float32)

def preprocess_mfcc_for_model(mfcc, img_size=IMG_SIZE):
    """Resize and normalize MFCC"""
    x = np.expand_dims(mfcc, axis=-1)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    x_resized = tf.image.resize(x_tf, img_size)
    mean = tf.reduce_mean(x_resized)
    std = tf.math.reduce_std(x_resized)
    x_norm = (x_resized - mean) / (std + 1e-8)
    return np.expand_dims(x_norm.numpy(), axis=0)

def load_best_model():
    """Load the saved model"""
    if os.path.exists(SAVE_MODEL_PATH):
        print("Loaded model from:", SAVE_MODEL_PATH)
        return load_model(SAVE_MODEL_PATH, compile=False)
    elif os.path.exists(LAST_CHECKPOINT):
        print("Loaded model from:", LAST_CHECKPOINT)
        return load_model(LAST_CHECKPOINT, compile=False)
    else:
        raise FileNotFoundError("No saved model found.")

# ---------- Prediction ----------
def predict_audio_file(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    classes = discover_classes(CLASS_FOLDER)
    model = load_best_model()
    
    mfcc = compute_mfcc_for_file(audio_path)
    x = preprocess_mfcc_for_model(mfcc)
    
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argmax(probs)
    
    print(f"\n🎧 Predictions for: {os.path.basename(audio_path)}\n")
    for i in np.argsort(probs)[::-1][:3]:
        print(f"  {classes[i]:20s} -> {probs[i]:.4f}")
    
    predicted_class = classes[top_idx]
    confidence = probs[top_idx]
    
    print(f"\n✅ Predicted Class: **{predicted_class}** (probability: {confidence:.4f})")

    # Optional MFCC visualization
    plt.figure(figsize=(8, 3))
    plt.title(f"MFCC Preview - Predicted: {predicted_class}")
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# ---------- Run ----------
if __name__ == "__main__":
    predict_audio_file(AUDIO_PATH)
