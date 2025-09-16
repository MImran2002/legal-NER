# training/distill_ner_tf.py — distill + export aligned to your structure
import os, json, subprocess, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, AutoConfig
from prepare_conll import read_conll

# --------------------------- Paths ---------------------------
HERE = Path(__file__).resolve().parent           # .../training
DATA_DIR = HERE / "data"                         # .../training/data
LABELS_TXT = HERE / "labels.txt"                 # .../training/labels.txt
WEB_DIR = HERE.parent / "web"                    # .../web
WEB_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_OUT = HERE / "mini_legalbert_ner_tf"     # student HF/Keras lives in training/
TFJS_OUT    = WEB_DIR / "model_mini_int8"        # TF.js export
TOK_OUT_WEB = WEB_DIR / "tokenizer"              # tokenizer for browser
TOK_OUT_WEB.mkdir(parents=True, exist_ok=True)

# --------------------------- Config ---------------------------
TEACHER_NAME = "nlpaueb/legal-bert-base-uncased"   # or str(HERE/'legalbert_ner_tf')
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 6
LR           = 3e-5
ALPHA        = 0.6
TEMP         = 2.0
SEED         = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------- Labels ----------
if not LABELS_TXT.exists():
    raise FileNotFoundError(f"Missing labels.txt at {LABELS_TXT}")
labels = [l.strip() for l in LABELS_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
num_labels = len(labels)

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME, use_fast=True)

def tokenize_and_align(tokens_list: List[List[str]], tags_list: List[List[str]]):
    enc = tokenizer(
        tokens_list,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=False,
    )
    aligned = []
    for i, tags in enumerate(tags_list):
        word_ids = enc.word_ids(batch_index=i)
        out = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                out.append(-100)
            elif wid != prev_word:
                out.append(label2id[tags[wid]])
            else:
                prev = tags[wid]
                if prev.startswith("B-") and ("I-"+prev[2:]) in label2id:
                    out.append(label2id["I-"+prev[2:]])
                else:
                    out.append(label2id[prev])
            prev_word = wid
        aligned.append(out)
    return enc, aligned

# ---------- Load data ----------
train_path = DATA_DIR / "train.conll"
dev_path   = DATA_DIR / "dev.conll"
if not train_path.exists() or not dev_path.exists():
    raise FileNotFoundError(f"Expected train/dev at {train_path} and {dev_path}")
train_tokens, train_tags = read_conll(str(train_path))
dev_tokens,   dev_tags   = read_conll(str(dev_path))
assert len(train_tokens) and len(dev_tokens), "Empty train/dev data."
assert len(train_tokens)==len(train_tags) and len(dev_tokens)==len(dev_tags)

train_enc, train_y = tokenize_and_align(train_tokens, train_tags)
dev_enc,   dev_y   = tokenize_and_align(dev_tokens,   dev_tags)

def ds_from(enc, y, bs):
    X = {
        "input_ids":      np.asarray(enc["input_ids"], dtype=np.int32),
        "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int32),
        "token_type_ids": np.asarray(enc.get("token_type_ids", np.zeros_like(enc["input_ids"])), dtype=np.int32),
    }
    Y = np.asarray(y, dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(2048, seed=SEED).batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = ds_from(train_enc, train_y, BATCH_SIZE)
dev_ds   = ds_from(dev_enc,   dev_y,   BATCH_SIZE)

# ---------- Teacher (frozen) ----------
teacher_cfg = AutoConfig.from_pretrained(
    TEACHER_NAME,
    num_labels=num_labels,
    id2label={str(i): l for i,l in id2label.items()},
    label2id=label2id,
)
teacher = TFAutoModelForTokenClassification.from_pretrained(
    TEACHER_NAME, config=teacher_cfg, from_pt=True
)
teacher.trainable = False

# ---------- Student (compact) ----------
base_cfg = AutoConfig.from_pretrained(TEACHER_NAME)
student_cfg = AutoConfig.from_pretrained(
    TEACHER_NAME,
    num_labels=num_labels,
    vocab_size=base_cfg.vocab_size,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    id2label={str(i): l for i,l in id2label.items()},
    label2id=label2id,
)
student = TFAutoModelForTokenClassification.from_config(student_cfg)

# ---------- Losses ----------
IGNORE_INDEX = -100
ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

def masked_ce(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, IGNORE_INDEX), tf.float32)
    y_true_safe = tf.where(tf.not_equal(y_true, IGNORE_INDEX), y_true, 0)
    loss_token = ce(y_true_safe, y_pred) * mask
    return tf.reduce_sum(loss_token) / tf.reduce_sum(mask)

def kd_loss_masked(student_logits, teacher_logits, y_true, T=2.0):
    mask = tf.cast(tf.not_equal(y_true, IGNORE_INDEX), tf.float32)
    s = tf.nn.log_softmax(student_logits / T, axis=-1)
    t = tf.nn.softmax(teacher_logits / T, axis=-1)
    kl = tf.reduce_sum(t * (tf.math.log(tf.maximum(t, 1e-8)) - s), axis=-1) * mask
    return (tf.reduce_sum(kl) / tf.reduce_sum(mask)) * (T*T)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def train_step(batch):
    x, y = batch
    with tf.GradientTape() as tape:
        s_logits = student(x, training=True).logits
        t_logits = teacher(x, training=False).logits
        loss_ce = masked_ce(y, s_logits)
        loss_kd = kd_loss_masked(s_logits, t_logits, y, T=TEMP)
        loss = ALPHA * loss_ce + (1.0 - ALPHA) * loss_kd
    grads = tape.gradient(loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return loss, loss_ce, loss_kd

@tf.function
def val_step(batch):
    x, y = batch
    s_logits = student(x, training=False).logits
    return masked_ce(y, s_logits)

# ---------- Training loop ----------
for epoch in range(1, EPOCHS+1):
    tr_losses = []
    for batch in train_ds:
        tr_losses.append(train_step(batch)[0])
    val_losses = [val_step(b) for b in dev_ds]
    print(f"Epoch {epoch}/{EPOCHS}  train {tf.reduce_mean(tr_losses):.4f}  val {tf.reduce_mean(val_losses):.4f}")

# ---------- Save student ----------
STUDENT_OUT.mkdir(parents=True, exist_ok=True)
student.save_pretrained(str(STUDENT_OUT))
tokenizer.save_pretrained(str(STUDENT_OUT))

sm_dir = STUDENT_OUT / "keras_saved_model"
tf.saved_model.save(student, str(sm_dir))

# ---------- Tokenizer → web/tokenizer ----------
TOK_OUT_WEB.mkdir(parents=True, exist_ok=True)
vocab = tokenizer.get_vocab()
(TOK_OUT_WEB / "vocab.json").write_text(json.dumps({
    "vocab": vocab,
    "unk_token": tokenizer.unk_token,
    "cls_token": tokenizer.cls_token,
    "sep_token": tokenizer.sep_token,
    "pad_token": tokenizer.pad_token,
    "mask_token": getattr(tokenizer, "mask_token", "[MASK]"),
    "do_lower_case": True
}, ensure_ascii=False), encoding="utf-8")

(TOK_OUT_WEB / "config.json").write_text(json.dumps(
    {"id2label": {str(i): id2label[i] for i in range(num_labels)},
     "label2id": label2id,
     "max_length": MAX_LEN}, indent=2), encoding="utf-8")

# Note: your frontend uses 'label.json' (singular)
(WEB_DIR / "label.json").write_text(json.dumps([id2label[i] for i in range(num_labels)]), encoding="utf-8")
print("✓ Student + tokenizer saved. Exporting TF.js …")

# ---------- Convert to TF.js INT8 ----------
TFJS_OUT.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-m", "tensorflowjs.converters.converter",
    "--input_format=tf_saved_model",
    "--signature_name=serving_default",
    "--saved_model_tags=serve",
    "--quantize_uint8",
    "--weight_shard_size_bytes=4194304",
    str(sm_dir),
    str(TFJS_OUT),
]
print("Running:", " ".join(cmd))
try:
    subprocess.run(cmd, check=True)
    print("✓ TF.js INT8 exported to", TFJS_OUT)
except Exception as e:
    print("⚠️ INT8 export via CLI failed:", e)
    print("Attempting non-quantized export instead...")
    cmd_fallback = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=tf_saved_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        "--weight_shard_size_bytes=4194304",
        str(sm_dir),
        str(TFJS_OUT),
    ]
    subprocess.run(cmd_fallback, check=True)
    print("✓ TF.js (no-quant) exported to", TFJS_OUT)
