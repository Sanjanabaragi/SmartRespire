# ---------- Imports ----------
import os
import math
import random
from pathlib import Path
from datetime import datetime
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Reshape, LayerNormalization, GlobalAveragePooling1D,
                                     Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd

# -------- USER----------
DRIVE_WORKDIR = "/content/drive/MyDrive/major/project/dataset"  # change if needed
OUT_MFCC_DIR = os.path.join(DRIVE_WORKDIR, "data/mfcc_augmented")

# model / data params (must match how MFCCs were computed)
IMG_SIZE = (128, 128)
N_MFCC = 40
MAX_PAD_LEN = 862
BATCH_SIZE = 8
EPOCHS = 30
RANDOM_SEED = 42

# Paths for saving models (native Keras format)
SAVE_MODEL_PATH = os.path.join(DRIVE_WORKDIR, "best_model.keras")
LAST_CHECKPOINT = os.path.join(DRIVE_WORKDIR, "last_model.keras")

# Toggle: if you want to resume from LAST_CHECKPOINT (if exists), set True
RESUME_FROM_LAST = False

# reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------- Basic checks ----------
if not os.path.exists(OUT_MFCC_DIR):
    raise SystemExit(f"OUT_MFCC_DIR not found: {OUT_MFCC_DIR}. Point to the folder you previously created.")

train_root = os.path.join(OUT_MFCC_DIR, "train")
val_root = os.path.join(OUT_MFCC_DIR, "val")

if not os.path.isdir(train_root):
    raise SystemExit(f"Train folder missing: {train_root}")

# ---------- Detect classes from train folders ----------
classes = sorted([d.name for d in Path(train_root).iterdir() if d.is_dir()])
if not classes:
    raise SystemExit(f"No class subfolders found under {train_root}")

print("Detected classes (from train folders):", classes)
num_classes = len(classes)

# ---------- Helper: gather file lists (accept any .npy) ----------
def gather_paths_per_class(root_dir, classes, pattern="*.npy"):
    paths_per_class = []
    counts = []
    for cls in classes:
        folder = os.path.join(root_dir, cls)
        if os.path.isdir(folder):
            files = sorted(glob(os.path.join(folder, pattern)))
        else:
            files = []
        paths_per_class.append(files)
        counts.append(len(files))
    return paths_per_class, counts

# NOTE: use "*.npy" so copied files with different names are discovered
train_paths_per_class, train_counts = gather_paths_per_class(train_root, classes, pattern="*.npy")
val_paths_per_class, val_counts = gather_paths_per_class(val_root, classes, pattern="*.npy") if os.path.isdir(val_root) else ([[] for _ in classes], [0]*num_classes)

print("Train counts per class:", dict(zip(classes, train_counts)))
print("Val counts per class:  ", dict(zip(classes, val_counts)))

if sum(train_counts) == 0:
    raise SystemExit("No training files found. Check OUT_MFCC_DIR/train/<class>/*.npy")

# ---------- LabelEncoder consistent with classes ----------
le = LabelEncoder()
le.fit(classes)

# ---------- TF dataset helpers ----------
AUTOTUNE = tf.data.AUTOTUNE

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
        x = tf.expand_dims(x, axis=-1)                  # (n_mfcc, time, 1)
        x = tf.image.resize(x, IMG_SIZE)                # resize to IMG_SIZE
        x = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x)
        std = tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-8)
        y = tf.one_hot(class_index, depth=num_classes)
        return x, y
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    return ds

# Build per-class datasets for training (balanced sampling)
class_ds_list = []
for i, fps in enumerate(train_paths_per_class):
    if len(fps) == 0:
        dummy_x = np.zeros(IMG_SIZE + (1,), dtype=np.float32)
        ds = tf.data.Dataset.from_tensors((dummy_x, tf.one_hot(i, depth=num_classes))).repeat()
        class_ds_list.append(ds)
    else:
        class_ds_list.append(make_class_dataset(fps, class_index=i, repeat=True))

balanced_train_ds = tf.data.experimental.sample_from_datasets(class_ds_list, weights=None, seed=RANDOM_SEED)
balanced_train_ds = balanced_train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

# Build validation dataset (if any)
val_files_flat = []
val_labels_int = []
for i, fps in enumerate(val_paths_per_class):
    for p in fps:
        val_files_flat.append(p)
        val_labels_int.append(i)

if val_files_flat:
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
else:
    val_ds = None
    val_labels_np = np.array([])

# steps calculations — recompute val_steps from actual flattened val list
largest_class = max(train_counts) if train_counts else 0
steps_per_epoch = math.ceil((largest_class * num_classes) / BATCH_SIZE)
val_steps = math.ceil(len(val_files_flat) / BATCH_SIZE) if val_files_flat else 0
print(f"steps_per_epoch={steps_per_epoch}, val_steps={val_steps}, total_val_files={len(val_files_flat)}")

# ---------- Model definition ----------
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

# Optionally resume from LAST_CHECKPOINT
if RESUME_FROM_LAST and os.path.exists(LAST_CHECKPOINT):
    print("Resuming model from last checkpoint:", LAST_CHECKPOINT)
    try:
        model = tf.keras.models.load_model(LAST_CHECKPOINT, compile=False)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Loaded and compiled last checkpoint.")
    except Exception as e:
        print("Failed to load/compile last checkpoint; building a fresh model. Error:", e)
        model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=num_classes)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
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
        self.true_labels_np = true_labels_np
        self.batch_size = batch_size
        self.best = -1.0
        self.save_best_path = save_best_path

    def on_epoch_end(self, epoch, logs=None):
        if self.val_ds is None or self.val_steps == 0:
            print(" — no validation available this epoch")
            return
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
                try:
                    self.model.save(self.save_best_path, include_optimizer=False)
                except Exception as e:
                    print("Error saving best model:", e)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-7)
val_f1_ds_cb = ValF1CallbackDS(val_ds=val_ds, val_steps=val_steps, true_labels_np=val_labels_np, batch_size=BATCH_SIZE, save_best_path=SAVE_MODEL_PATH)

# ---------- Train ----------
start_time = datetime.now()
callbacks = [reduce_lr, val_f1_ds_cb]
try:
    if val_ds is not None and val_steps > 0:
        history = model.fit(
            balanced_train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("No validation files found — training without validation.")
        history = model.fit(
            balanced_train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
except tf.errors.ResourceExhaustedError as e:
    print("OOM: reduce BATCH_SIZE or IMG_SIZE, or use fewer model params. Error:")
    print(e)
    raise

print("Training finished in:", datetime.now() - start_time)

# Save last checkpoint
try:
    model.save(LAST_CHECKPOINT, include_optimizer=False)
    print("Saved last checkpoint to:", LAST_CHECKPOINT)
except Exception as e:
    print("Error saving last checkpoint:", e)

# ---------- Evaluate final / best model (robust to missing val classes) ----------
best_model = model
if os.path.exists(SAVE_MODEL_PATH):
    print("Loading best model from:", SAVE_MODEL_PATH)
    try:
        best_model = tf.keras.models.load_model(SAVE_MODEL_PATH, compile=False)
        print("Loaded best model successfully.")
    except Exception as e:
        print("Failed to load best model; using in-memory model. Error while loading:", e)
else:
    print("No saved best model found — using current model.")

# Evaluate only if val exists
if val_ds is not None and val_steps > 0:
    preds = best_model.predict(val_ds, steps=val_steps, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = val_labels_np[:len(pred_labels)]

    # save predictions & truths
    np.savetxt(os.path.join(DRIVE_WORKDIR, "y_true.txt"), true_labels, fmt='%d')
    np.savetxt(os.path.join(DRIVE_WORKDIR, "y_pred.txt"), pred_labels, fmt='%d')

    print("\nClassification Report (validation):")
    # Force report to include all classes (0..num_classes-1) so target_names aligns.
    all_label_indices = list(range(num_classes))
    print(classification_report(true_labels,
                                pred_labels,
                                labels=all_label_indices,
                                target_names=list(le.classes_),
                                zero_division=0))

    cm = confusion_matrix(true_labels, pred_labels, labels=all_label_indices)
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

    # per-class metrics CSV (ensures same label indexing)
    p, r, f, s = precision_recall_fscore_support(true_labels,
                                                 pred_labels,
                                                 labels=all_label_indices,
                                                 zero_division=0)
    df_metrics = pd.DataFrame({
        "class_index": all_label_indices,
        "class_name": list(le.classes_),
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s
    })
    metrics_csv = os.path.join(DRIVE_WORKDIR, "per_class_metrics_val.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print("Saved per-class metrics to:", metrics_csv)
else:
    print("No validation dataset available — skipping evaluation & metrics saving.")

print("Done. Artifacts (models/metrics) will be saved under:", DRIVE_WORKDIR)
