import os
import argparse
import csv
import numpy as np
import tensorflow as tf
import librosa
import sys
from typing import List

# tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as e:
    tk = None

# ---------------- USER----------------
DEFAULT_MODEL = "E:/colab/best_model.keras"          
FALLBACK_MODEL ="E:/colab/last_model.keras"
# Manually define classes in the same order used for training:
CLASSES = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",
    "Healthy",
    "LRTI",
    "Pneumonia",
    "URTI"
]
IMG_SIZE = (128, 128)
N_MFCC = 40
MAX_PAD_LEN = 862
SR = 22050
TOP_K_DEFAULT = 3
# -------------------------------------------------------------------

def load_model_try(paths: List[str]):
    for p in paths:
        if p and os.path.exists(p):
            print(f"Loading model from: {p}")
            try:
                m = tf.keras.models.load_model(p, compile=False)
                print("Model loaded.")
                return m
            except Exception as e:
                print(f"Failed loading model at {p}: {e}")
    raise FileNotFoundError(f"No valid model found in: {paths}")

def compute_mfcc_padtrim(y, sr, n_mfcc=N_MFCC, max_len=MAX_PAD_LEN):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    elif mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    return mfcc.astype(np.float32)

def preprocess_mfcc_for_model(mfcc_2d, img_size=IMG_SIZE):
    x = np.expand_dims(mfcc_2d, axis=-1)  # (n_mfcc, time, 1)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    x_resized = tf.image.resize(x_tf, size=img_size)
    x_np = x_resized.numpy()
    mean = x_np.mean()
    std = x_np.std()
    x_norm = (x_np - mean) / (std + 1e-8)
    x_batched = np.expand_dims(x_norm, axis=0)  # (1, H, W, 1)
    return x_batched

def predict_file(audio_path: str, model, classes: List[str], sr=SR, top_k=TOP_K_DEFAULT):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    print(f"\nProcessing file: {audio_path}")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = compute_mfcc_padtrim(y, sr=sr)
    x = preprocess_mfcc_for_model(mfcc)
    preds = model.predict(x, verbose=0)
    probs = preds[0]
    top_idx = probs.argsort()[::-1][:top_k]
    results = [(classes[i], float(probs[i]), int(i)) for i in top_idx]
    return results, probs

def choose_files_via_gui(initialdir=None):
    if tk is None:
        raise RuntimeError("tkinter not available on this Python environment.")
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
    filenames = filedialog.askopenfilenames(title="Select audio file(s) for prediction",
                                            initialdir=initialdir or os.getcwd(),
                                            filetypes=filetypes)
    root.destroy()
    return list(filenames)

def main():
    parser = argparse.ArgumentParser(description="GUI audio prediction script (local)")
    parser.add_argument("--model", help="Path to .keras model", default=None)
    parser.add_argument("--topk", type=int, help="Top-K predictions", default=TOP_K_DEFAULT)
    parser.add_argument("--out-csv", help="Optional CSV path to save results", default=None)
    parser.add_argument("--nogui", action="store_true", help="Do not open GUI; instead pass audio paths after -- (positional args)")
    parser.add_argument("audio", nargs="*", help="Audio file(s) (only used when --nogui specified)")
    args = parser.parse_args()

    model_paths = [args.model] if args.model else [DEFAULT_MODEL, FALLBACK_MODEL]
    model = load_model_try(model_paths)

    classes = CLASSES
    if model.output_shape[-1] != len(classes):
        print(f"Warning: model has output size {model.output_shape[-1]} but you supplied {len(classes)} class names.")
        # continue anyway

    if args.nogui:
        if not args.audio:
            print("No audio files provided on command line. Exiting.")
            sys.exit(1)
        audio_files = args.audio
    else:
        try:
            audio_files = choose_files_via_gui()
        except Exception as e:
            print("Failed to open GUI file dialog:", e)
            print("You can run with --nogui and pass file paths instead.")
            sys.exit(1)

    if not audio_files:
        print("No files selected. Exiting.")
        sys.exit(0)

    rows = []
    for f in audio_files:
        try:
            results, probs = predict_file(f, model, classes, top_k=args.topk)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

        print("\nPredictions (top {}):".format(args.topk))
        for name, p, idx in results:
            print(f"  - {name:<20s} (class_idx={idx}) : {p:.6f}")

        print("\nFull probabilities by class:")
        for i, name in enumerate(classes):
            prob_i = float(probs[i]) if i < len(probs) else 0.0
            print(f"  [{i}] {name:<20s} : {prob_i:.6f}")

        # prepare CSV row
        row = {"file": f}
        for i, name in enumerate(classes):
            row[f"p_{i}_{name}"] = float(probs[i]) if i < len(probs) else 0.0
        # top prediction
        top0 = results[0] if results else ("", 0.0, -1)
        row["pred_label"] = top0[0]
        row["pred_prob"] = top0[1]
        rows.append(row)

    # save CSV if requested
    if args.out_csv:
        csv_path = args.out_csv
        # collect header
        fieldnames = ["file", "pred_label", "pred_prob"] + [f"p_{i}_{name}" for i, name in enumerate(classes)]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"\nSaved results to CSV: {csv_path}")
        except Exception as e:
            print("Failed to save CSV:", e)

    print("\nDone.")

if __name__ == "__main__":
    main()
