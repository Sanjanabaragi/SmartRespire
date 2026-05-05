# Colab-ready full training script (balanced per-class augmentation)
# 1) Set Runtime -> Change runtime type -> GPU
# 2) Paste & run this single cell.

# ---------- Install deps & mount Drive ----------
!pip install -q librosa soundfile

# ---------- Imports ----------
import os
import random
from pathlib import Path
from datetime import datetime
import math
import numpy as np
import pandas as pd
import librosa
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Reshape, LayerNormalization, GlobalAveragePooling1D,
                                     Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support

# ---------- GPU safety / mixed precision (optional) ----------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected. Memory growth enabled.")
        # Optional: enable mixed precision (speeds up on many GPUs)
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled.")
        except Exception as e:
            print("Mixed precision not enabled:", e)
    except RuntimeError as e:
        print("Could not set GPU memory growth:", e)
else:
    print("No GPU detected — training will use CPU.")

# ---------- User config: change DRIVE_WORKDIR to a Drive folder you control ----------
DRIVE_WORKDIR = "/content/drive/MyDrive/major/project/dataset"  # <- change if you like
os.makedirs(DRIVE_WORKDIR, exist_ok=True)

AUDIO_DIR = os.path.join(DRIVE_WORKDIR, "ICBHI_final_database")  # put .wav files here
DIAG_CSV = os.path.join(DRIVE_WORKDIR, "patient_diagnosis.csv")  # patient diag CSV
OUT_MFCC_DIR = os.path.join(DRIVE_WORKDIR, "data/mfcc_augmented")       # precomputed MFCCs & orig audio

# audio / MFCC / training params (tweak as needed)
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 20
N_MFCC = 40
MAX_PAD_LEN = 862
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 5
RANDOM_SEED = 42
RARE_AUG_PER_CLIP = 5
OTHER_AUG_PER_CLIP = 2
RARE_CLASSES = {"Asthma", "LRTI"}
ALL_AUG_FOR_ALL_CLASSES = True
SAVE_MODEL_PATH = os.path.join(DRIVE_WORKDIR, "best_model.h5")
LAST_CHECKPOINT = os.path.join(DRIVE_WORKDIR, "last_model.h5")

# NEW: target per-class. If None -> auto set to max class count after saving originals.
TARGET_PER_CLASS = None  # e.g. set to 200 to force exactly 200 per class (train only), or leave None

# reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs(OUT_MFCC_DIR, exist_ok=True)

# ---------- Helper utilities ----------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_patient_diag(diag_csv):
    df = pd.read_csv(diag_csv, header=None)
    df.columns = ["patient_id", "diagnosis"]
    return df

def segment_audio(audio, sr, segment_seconds=20):
    seg_len = int(segment_seconds * sr)
    segments = []
    total = len(audio)
    if total < seg_len:
        padded = np.pad(audio, (0, seg_len - total), mode='constant')
        segments.append(padded)
        return segments
    start = 0
    while start + seg_len <= total:
        segments.append(audio[start:start + seg_len])
        start += seg_len
    return segments

def augment_noise(audio, scale=0.005):
    noise = np.random.randn(len(audio)) * scale
    return audio + noise

def augment_pitch(audio, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)
    except Exception:
        return audio

def augment_time_stretch(audio, rate):
    try:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        target_len = int(SEGMENT_SECONDS * SAMPLE_RATE)
        if len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)), mode='constant')
        else:
            stretched = stretched[:target_len]
        return stretched
    except Exception:
        return audio

def compute_mfcc(audio, sr, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.astype(np.float32)

def save_mfcc(mfcc, out_path):
    np.save(out_path, mfcc)

# ---------- Prepare file list & labels ----------
print("Checking dataset paths...")
if not os.path.exists(AUDIO_DIR):
    raise SystemExit(f"AUDIO_DIR not found: {AUDIO_DIR}. Put your wavs there (in Drive) or update AUDIO_DIR.")
if not os.path.exists(DIAG_CSV):
    raise SystemExit(f"DIAG_CSV not found: {DIAG_CSV}. Put patient_diagnosis.csv there or update DIAG_CSV.")

print("Loading file list and labels...")
filenames = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav') and os.path.isfile(os.path.join(AUDIO_DIR, f))]
if not filenames:
    raise SystemExit(f"No wav files found in {AUDIO_DIR}")

p_ids = [int(f[:3]) for f in filenames]
filepaths = [os.path.join(AUDIO_DIR, f) for f in filenames]

p_diag_df = load_patient_diag(DIAG_CSV)
pid_to_diag = dict(zip(p_diag_df["patient_id"], p_diag_df["diagnosis"]))

labels = []
for pid in p_ids:
    if pid not in pid_to_diag:
        raise SystemExit(f"Patient id {pid} not found in {DIAG_CSV}")
    labels.append(pid_to_diag[pid])

labels = np.array(labels)
p_ids = np.array(p_ids)
filepaths = np.array(filepaths)

unique_patients = np.unique(p_ids)
train_patients, val_patients = train_test_split(unique_patients, test_size=0.2, random_state=RANDOM_SEED)
print(f"Total patients: {len(unique_patients)} | Train patients: {len(train_patients)} | Val patients: {len(val_patients)}")

# create class folders in Drive (for mfccs and orig_audio)
for split in ["train", "val"]:
    for cls in np.unique(labels):
        ensure_dir(os.path.join(OUT_MFCC_DIR, split, cls))
# folder for saving original audio segments to augment from later
for split in ["train", "val"]:
    for cls in np.unique(labels):
        ensure_dir(os.path.join(OUT_MFCC_DIR, "orig_audio", split, cls))

# ---------- Precompute ORIGINAL MFCCs & save original segments (no augment yet) ----------
# This first pass saves only the original (per-segment) MFCCs and also saves the raw audio segments as .npy.
# Later we'll determine the target per-class count and run augmentation to reach the target (train only).

print("Precomputing ORIGINAL MFCCs and saving original audio segments (skips existing files)...")

# counters start from existing files to avoid overwrite
save_counters = {}
for split in ["train","val"]:
    for cls in np.unique(labels):
        folder = Path(os.path.join(OUT_MFCC_DIR, split, cls))
        existing = list(folder.glob("mfcc_*.npy"))
        save_counters[(split, cls)] = len(existing)

# also counters for original audio saved
audio_save_counters = {}
for split in ["train","val"]:
    for cls in np.unique(labels):
        folder = Path(os.path.join(OUT_MFCC_DIR, "orig_audio", split, cls))
        existing = list(folder.glob("seg_*.npy"))
        audio_save_counters[(split, cls)] = len(existing)

for fp, pid, cls in tqdm(zip(filepaths, p_ids, labels), total=len(filepaths), desc="Files (originals)"):
    try:
        audio, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading {fp}: {e}")
        continue

    segments = segment_audio(audio, SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS)
    split = "train" if pid in train_patients else "val"

    for seg in segments:
        # save MFCC original if missing
        idx = save_counters[(split, cls)]
        out_path = os.path.join(OUT_MFCC_DIR, split, cls, f"mfcc_{idx:06d}.npy")
        if not os.path.exists(out_path):
            mfcc_orig = compute_mfcc(seg, SAMPLE_RATE)
            save_mfcc(mfcc_orig, out_path)
        save_counters[(split, cls)] += 1

        # save corresponding raw audio segment for later augmentation
        idx_a = audio_save_counters[(split, cls)]
        audio_out = os.path.join(OUT_MFCC_DIR, "orig_audio", split, cls, f"seg_{idx_a:06d}.npy")
        if not os.path.exists(audio_out):
            np.save(audio_out, seg.astype(np.float32))
        audio_save_counters[(split, cls)] += 1

print("Original precomputation complete. Summary per class & split (originals):")
for split in ["train", "val"]:
    for cls in np.unique(labels):
        d = Path(os.path.join(OUT_MFCC_DIR, split, cls))
        count = len(list(d.glob("mfcc_*.npy")))
        print(f"  {split}/{cls}: {count} files (originals)")

# ---------- Decide target per-class (train only) ----------
# If TARGET_PER_CLASS is None -> set to max current training class count (i.e., upsample smaller classes to largest)
# You can override TARGET_PER_CLASS above to set a custom limit.
train_counts_current = {cls: len(list(Path(os.path.join(OUT_MFCC_DIR, "train", cls)).glob("mfcc_*.npy"))) for cls in np.unique(labels)}
print("Current train counts (per class):", train_counts_current)

if TARGET_PER_CLASS is None:
    TARGET_PER_CLASS = max(train_counts_current.values())
    print(f"Auto TARGET_PER_CLASS set to largest class count: {TARGET_PER_CLASS}")
else:
    print(f"TARGET_PER_CLASS manually set to: {TARGET_PER_CLASS}")

# ---------- Augmentation pass: augment training classes until each reaches TARGET_PER_CLASS ----------
# We will load saved original raw segment .npy files from OUT_MFCC_DIR/orig_audio/train/<cls>/seg_*.npy
# and apply augment ops (same ops as before). For val we do not augment.

print("Starting augmentation pass to balance train classes...")

# augmentation ops list (same as before)
base_aug_ops = []
base_aug_ops.append(("noise",))
base_aug_ops.append(("pitch_pos",))
base_aug_ops.append(("pitch_neg",))
base_aug_ops.append(("stretch_slow",))
base_aug_ops.append(("stretch_fast",))
base_aug_ops.extend([("noise","pitch_pos"), ("noise","pitch_neg"), ("noise","stretch_slow"), ("noise","stretch_fast"),
                     ("pitch_pos","stretch_slow"), ("pitch_pos","stretch_fast"), ("pitch_neg","stretch_slow"), ("pitch_neg","stretch_fast")])
base_aug_ops.extend([("noise","pitch_pos","stretch_slow"), ("noise","pitch_neg","stretch_fast")])

# helper: load list of available original segment files per class (train)
orig_audio_paths_per_class = {}
for cls in np.unique(labels):
    p = sorted(glob(os.path.join(OUT_MFCC_DIR, "orig_audio", "train", cls, "seg_*.npy")))
    orig_audio_paths_per_class[cls] = p

# counters recompute for accurate state
for cls in np.unique(labels):
    save_counters[("train", cls)] = len(list(Path(os.path.join(OUT_MFCC_DIR, "train", cls)).glob("mfcc_*.npy")))

# Augment until each class reaches TARGET_PER_CLASS
for cls in tqdm(sorted(np.unique(labels)), desc="Augment classes"):
    current = save_counters[("train", cls)]
    if current >= TARGET_PER_CLASS:
        continue  # already at or above target
    orig_paths = orig_audio_paths_per_class.get(cls, [])
    if len(orig_paths) == 0:
        print(f"WARNING: no original audio segments found for class {cls}. Cannot augment.")
        continue

    # We'll cycle through original segments and apply random augment ops until target reached
    k = 0
    while save_counters[("train", cls)] < TARGET_PER_CLASS:
        # pick a source original segment (cycle through)
        src_idx = k % len(orig_paths)
        src_path = orig_paths[src_idx]
        try:
            seg = np.load(src_path)
        except Exception as e:
            print(f"Error loading orig segment {src_path}: {e}")
            k += 1
            continue

        # decide augmentation ops:
        # if ALL_AUG_FOR_ALL_CLASSES is True -> use same aug choices
        # else use more augmentations for rare classes (RARE_CLASSES)
        if ALL_AUG_FOR_ALL_CLASSES:
            aug_candidates = base_aug_ops.copy()
        else:
            aug_candidates = base_aug_ops.copy() if cls in RARE_CLASSES else [("noise",), ("pitch_pos",), ("pitch_neg",)]

        # pick one augmentation combination at random
        ops = random.choice(aug_candidates)
        # create augmented audio
        aug_audio = seg.copy()
        for op in ops:
            if op == "noise":
                aug_audio = augment_noise(aug_audio, scale=0.005)
            elif op == "pitch_pos":
                aug_audio = augment_pitch(aug_audio, SAMPLE_RATE, n_steps=1)
            elif op == "pitch_neg":
                aug_audio = augment_pitch(aug_audio, SAMPLE_RATE, n_steps=-1)
            elif op == "stretch_slow":
                aug_audio = augment_time_stretch(aug_audio, rate=0.9)
            elif op == "stretch_fast":
                aug_audio = augment_time_stretch(aug_audio, rate=1.1)

        # compute mfcc and save
        idx2 = save_counters[("train", cls)]
        out_path2 = os.path.join(OUT_MFCC_DIR, "train", cls, f"mfcc_{idx2:06d}.npy")
        # if file exists (race condition), skip
        if os.path.exists(out_path2):
            save_counters[("train", cls)] += 1
            k += 1
            continue
        mfcc_aug = compute_mfcc(aug_audio, SAMPLE_RATE)
        save_mfcc(mfcc_aug, out_path2)
        save_counters[("train", cls)] += 1

        k += 1

    print(f"Class {cls}: reached {save_counters[('train', cls)]} files (target {TARGET_PER_CLASS})")

print("Augmentation pass complete. New train counts:")
for cls in np.unique(labels):
    c = len(list(Path(os.path.join(OUT_MFCC_DIR, "train", cls)).glob("mfcc_*.npy")))
    print(f"  train/{cls}: {c}")

# ---------- Build label encoder and class info ----------
le = LabelEncoder()
# Fit on classes present in your labels (train+val)
all_classes_sorted = sorted(np.unique(labels))
le.fit(all_classes_sorted)
print("Classes:", le.classes_)
num_classes = len(le.classes_)

# ---------- Balanced streaming TF Dataset (uses .npy files saved above) ----------
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(RANDOM_SEED)

def gather_paths_and_counts(split):
    classes = list(le.classes_)
    paths_per_class = []
    counts = []
    for cls in classes:
        folder = os.path.join(OUT_MFCC_DIR, split, cls)
        files = sorted(glob(os.path.join(folder, "mfcc_*.npy")))
        paths_per_class.append(files)
        counts.append(len(files))
    return classes, paths_per_class, counts

# numpy loader used via tf.numpy_function
def _np_load_npy(path_bytes):
    path = path_bytes.decode('utf-8')
    arr = np.load(path).astype(np.float32)
    return arr

def make_class_dataset(filepaths, class_index, repeat=True, shuffle_buffer=1024):
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    if shuffle_buffer and len(filepaths) > 1:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(filepaths)), seed=RANDOM_SEED)
    if repeat:
        ds = ds.repeat()
    def _load_and_preprocess(path):
        x = tf.numpy_function(_np_load_npy, [path], tf.float32)
        x.set_shape([N_MFCC, MAX_PAD_LEN])
        x = tf.expand_dims(x, axis=-1)
        x = tf.image.resize(x, IMG_SIZE)
        x = tf.cast(x, tf.float32)
        # per-sample standardization
        mean = tf.reduce_mean(x)
        std = tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-8)
        y = tf.one_hot(class_index, depth=num_classes)
        return x, y
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    return ds

# gather training and validation file lists
classes, train_paths_per_class, train_counts = gather_paths_and_counts("train")
val_classes, val_paths_per_class, val_counts = gather_paths_and_counts("val")

print("Train files per class:", dict(zip(classes, train_counts)))
print("Val files per class:", dict(zip(val_classes, val_counts)))

# build per-class datasets
class_ds_list = []
for i, fps in enumerate(train_paths_per_class):
    if len(fps) == 0:
        # yield a dummy constant if class missing (shouldn't happen)
        dummy_x = np.zeros(IMG_SIZE + (1,), dtype=np.float32)
        class_ds_list.append(tf.data.Dataset.from_tensors((dummy_x, tf.one_hot(i, depth=num_classes))).repeat())
    else:
        class_ds_list.append(make_class_dataset(fps, class_index=i, repeat=True))

# sample equally from each class dataset -> balanced stream
balanced_train_ds = tf.data.experimental.sample_from_datasets(class_ds_list, weights=None, seed=RANDOM_SEED)
balanced_train_ds = balanced_train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

# build validation dataset (deterministic, no repeat)
val_files_flat = []
val_labels_int = []
for i, fps in enumerate(val_paths_per_class):
    for p in fps:
        val_files_flat.append(p)
        val_labels_int.append(i)
val_files_flat = list(val_files_flat)
val_labels_np = np.array(val_labels_int, dtype=np.int32)

def make_eval_dataset(filepaths, labels_int):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels_int))
    def _load_val(path, lab):
        x = tf.numpy_function(_np_load_npy, [path], tf.float32)
        x.set_shape([N_MFCC, MAX_PAD_LEN])
        x = tf.expand_dims(x, axis=-1)
        x = tf.image.resize(x, IMG_SIZE)
        x = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x); std = tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-8)
        y = tf.one_hot(lab, depth=num_classes)
        return x, y
    ds = ds.map(_load_val, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

val_ds = make_eval_dataset(val_files_flat, val_labels_np)

# choose steps_per_epoch: ensure each class contributes approx equally per epoch
largest_class = max(train_counts) if len(train_counts) else 0
if largest_class == 0:
    raise SystemExit("No training files found — check OUT_MFCC_DIR/train/*/mfcc_*.npy")
steps_per_epoch = math.ceil((largest_class * len(classes)) / BATCH_SIZE)
val_steps = math.ceil(len(val_files_flat) / BATCH_SIZE)
print(f"steps_per_epoch={steps_per_epoch}, val_steps={val_steps}")

# ---------- Model (Deeper CNN + Transformer) ----------
def build_model(input_shape=(128,128,1), num_classes=6):
    inp = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # reshape for transformer
    shape = tf.keras.backend.int_shape(x)
    seq_len = shape[1] * shape[2]
    features = shape[3]
    x = Reshape((seq_len, features))(x)

    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = x + attn
    x = LayerNormalization()(x)
    x_ff = tf.keras.layers.Dense(256, activation='relu')(x)
    x = x + x_ff
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = Model(inputs=inp, outputs=out)
    return model

model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=num_classes)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- Callback: val macro-F1 that uses val_labels_np for true labels ----------
class ValF1CallbackDS(Callback):
    def __init__(self, val_ds, val_steps, true_labels_np, batch_size=32, save_best_path=None):
        super().__init__()
        self.val_ds = val_ds
        self.val_steps = val_steps
        self.true_labels_np = true_labels_np  # flattened ints in the same order used by val_ds iteration
        self.batch_size = batch_size
        self.best = -1.0
        self.save_best_path = save_best_path

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, steps=self.val_steps, verbose=0)
        preds_labels = np.argmax(preds, axis=1)
        true_labels = self.true_labels_np[:len(preds_labels)]
        f1 = f1_score(true_labels, preds_labels, average='macro', zero_division=0)
        logs = logs or {}
        logs['val_macro_f1'] = f1
        print(f" — val_macro_f1: {f1:.4f}")
        if self.save_best_path:
            if f1 > self.best:
                self.best = f1
                print(f"Saving improved model (val_macro_f1: {f1:.4f}) to {self.save_best_path}")
                self.model.save(self.save_best_path)

# ---------- Callbacks ----------
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-7)
val_f1_ds_cb = ValF1CallbackDS(val_ds=val_ds, val_steps=val_steps, true_labels_np=val_labels_np, batch_size=BATCH_SIZE, save_best_path=SAVE_MODEL_PATH)

# ---------- Train ----------
start_time = datetime.now()
try:
    history = model.fit(
        balanced_train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[reduce_lr, val_f1_ds_cb],
        verbose=1
    )
except tf.errors.ResourceExhaustedError as e:
    print("OOM: reduce BATCH_SIZE or IMG_SIZE, or use fewer model params. Error:")
    print(e)
    raise

print("Training finished in:", datetime.now() - start_time)
model.save(LAST_CHECKPOINT)
print("Saved last checkpoint to:", LAST_CHECKPOINT)

# ---------- Evaluate final / best model ----------
if os.path.exists(SAVE_MODEL_PATH):
    print("Loading best model from:", SAVE_MODEL_PATH)
    best_model = tf.keras.models.load_model(SAVE_MODEL_PATH, compile=False)
else:
    print("No saved best model found — using current model.")
    best_model = model

preds = best_model.predict(val_ds, steps=val_steps, verbose=1)
pred_labels = np.argmax(preds, axis=1)
true_labels = val_labels_np[:len(pred_labels)]

# save predictions & truths
np.savetxt(os.path.join(DRIVE_WORKDIR, "y_true.txt"), true_labels, fmt='%d')
np.savetxt(os.path.join(DRIVE_WORKDIR, "y_pred.txt"), pred_labels, fmt='%d')

print("\nClassification Report (validation):")
print(classification_report(true_labels, pred_labels, target_names=le.classes_, zero_division=0))
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion matrix:\n", cm)

# save confusion matrix image
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix (validation)")
plt.colorbar()
plt.xticks(range(num_classes), le.classes_, rotation=45, ha='right')
plt.yticks(range(num_classes), le.classes_)
th = cm.max()/2. if cm.max() > 0 else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i,j], 'd'), ha='center',
                 color='white' if cm[i,j] > th else 'black')
plt.tight_layout()
cm_path = os.path.join(DRIVE_WORKDIR, "confusion_matrix_val.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print("Saved confusion matrix to:", cm_path)

# per-class metrics CSV
p, r, f, s = precision_recall_fscore_support(true_labels, pred_labels, labels=list(range(num_classes)), zero_division=0)
df_metrics = pd.DataFrame({
    "class_index": list(range(num_classes)),
    "class_name": le.classes_,
    "precision": p,
    "recall": r,
    "f1": f,
    "support": s
})
metrics_csv = os.path.join(DRIVE_WORKDIR, "per_class_metrics_val.csv")
df_metrics.to_csv(metrics_csv, index=False)
print("Saved per-class metrics to:", metrics_csv)

print("All done. Artifacts saved under:", DRIVE_WORKDIR)
print("Tips: If you hit OOM, try BATCH_SIZE=4 (or 2), reduce IMG_SIZE, or switch to a generator/model with fewer params.")
