# training/train_legalbert_ner_tf.py
import os, json, math, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, AutoConfig

from prepare_conll import read_conll

# --------------------------- Paths ---------------------------
HERE = Path(__file__).resolve().parent          # .../training
DATA_DIR = HERE / "data"                        # .../training/data
LABELS_TXT = HERE / "labels.txt"                # .../training/labels.txt

# --------------------------- CLI ---------------------------
p = argparse.ArgumentParser()
p.add_argument("--model_name", default="nlpaueb/legal-bert-base-uncased")
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--output_dir", default=str(HERE / "legalbert_ner_tf"))  # stays in training/
p.add_argument("--seed", type=int, default=42)
p.add_argument("--prune", action="store_true", help="Enable magnitude pruning during training.")
p.add_argument("--final_sparsity", type=float, default=0.5, help="Target sparsity if --prune is set.")
p.add_argument("--begin_frac", type=float, default=0.0, help="Fraction of total steps when pruning begins.")
p.add_argument("--end_frac", type=float, default=0.8, help="Fraction of total steps when target sparsity is reached.")
args = p.parse_args()

MODEL_NAME = args.model_name
MAX_LEN    = args.max_len
BATCH_SIZE = args.batch_size
EPOCHS     = args.epochs
LR         = args.lr
OUTPUT_DIR = Path(args.output_dir)
SEED       = args.seed

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --------------------------- Labels ---------------------------
if not LABELS_TXT.exists():
    raise FileNotFoundError(f"Missing labels.txt at {LABELS_TXT}")
labels = [l.strip() for l in LABELS_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={str(i): l for i, l in id2label.items()},
    label2id=label2id,
)

# --------------------------- Data ---------------------------
train_path = DATA_DIR / "train.conll"
dev_path   = DATA_DIR / "dev.conll"
if not train_path.exists() or not dev_path.exists():
    raise FileNotFoundError(f"Expected train/dev at {train_path} and {dev_path}")

train_tokens, train_tags = read_conll(str(train_path))
dev_tokens,   dev_tags   = read_conll(str(dev_path))

assert len(train_tokens) > 0 and len(dev_tokens) > 0, "Empty train/dev."
assert len(train_tokens) == len(train_tags)
assert len(dev_tokens)   == len(dev_tags)

for i, (toks, tags) in enumerate(zip(train_tokens, train_tags)):
    if len(toks) != len(tags):
        raise ValueError(f"Train sent {i}: {len(toks)} tokens vs {len(tags)} tags.")
for i, (toks, tags) in enumerate(zip(dev_tokens, dev_tags)):
    if len(toks) != len(tags):
        raise ValueError(f"Dev sent {i}: {len(toks)} tokens vs {len(tags)} tags.")

def tokenize_and_align_labels(tokens_list: List[List[str]], tags_list: List[List[str]]):
    enc = tokenizer(
        tokens_list,
        is_split_into_words=True,
        return_offsets_mapping=False,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    labels_aligned = []
    for i, lab in enumerate(tags_list):
        word_ids = enc.word_ids(batch_index=i)
        label_ids, prev_word = [], None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(label2id[lab[wid]])
            else:
                prev_label = lab[wid]
                if prev_label.startswith("B-"):
                    label_ids.append(label2id.get("I-"+prev_label[2:], label2id[prev_label]))
                else:
                    label_ids.append(label2id[prev_label])
            prev_word = wid
        labels_aligned.append(label_ids)
    return enc, labels_aligned

train_enc, train_y = tokenize_and_align_labels(train_tokens, train_tags)
dev_enc,   dev_y   = tokenize_and_align_labels(dev_tokens, dev_tags)

def to_tf_dataset(enc, labels, bs):
    input_ids      = np.asarray(enc["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(enc["attention_mask"], dtype=np.int32)
    tti            = np.asarray(enc.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int32)
    y              = np.asarray(labels, dtype=np.int32)

    assert input_ids.shape == attention_mask.shape == tti.shape
    assert y.shape == input_ids.shape

    X = {"input_ids": input_ids, "token_type_ids": tti, "attention_mask": attention_mask}
    return tf.data.Dataset.from_tensor_slices((X, y)).shuffle(2048, seed=SEED).batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = to_tf_dataset(train_enc, train_y, BATCH_SIZE)
dev_ds   = to_tf_dataset(dev_enc,   dev_y,   BATCH_SIZE)

# --------------------------- Model ---------------------------
model = TFAutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, config=config, from_pt=True
)

# ----- Optional: PRUNING (wrap BEFORE compile) -----
callbacks = []
if args.prune:
    try:
        import tensorflow_model_optimization as tfmot
    except Exception as e:
        raise RuntimeError("Install tensorflow-model-optimization to use --prune") from e

    steps_per_epoch = int(math.ceil(len(train_enc["input_ids"]) / float(BATCH_SIZE)))
    begin_step = int(steps_per_epoch * EPOCHS * args.begin_frac)
    end_step   = int(steps_per_epoch * EPOCHS * args.end_frac)

    prune_sched = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=float(args.final_sparsity),
        begin_step=begin_step,
        end_step=end_step
    )
    model = tfmot.sparsity.keras.prune_low_magnitude(model, prune_schedule=prune_sched)
    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
    print(f"[prune] enabled: final_sparsity={args.final_sparsity}, steps {begin_step}->{end_step}")

# --------------------------- Compile/Train ---------------------------
base_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
def masked_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -100), tf.float32)
    loss_ = base_loss(y_true, y_pred) * mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="token_acc")]
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

model.compile(optimizer=optimizer, loss=masked_loss, metrics=metrics)
model.fit(train_ds, validation_data=dev_ds, epochs=EPOCHS, callbacks=callbacks)

# ----- Strip pruning masks BEFORE saving -----
if args.prune:
    import tensorflow_model_optimization as tfmot
    model = tfmot.sparsity.keras.strip_pruning(model)
    print("[prune] stripped pruning wrappers for export")

# --------------------------- Save ---------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tokenizer").mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

keras_export_dir = OUTPUT_DIR / "keras_saved_model"
tf.saved_model.save(model, str(keras_export_dir))
model.save(str(OUTPUT_DIR / "model.keras"))  # Keras v3 format

with open(OUTPUT_DIR / "tokenizer" / "config.json", "w", encoding="utf-8") as f:
    json.dump({"id2label": {str(i): id2label[i] for i in range(num_labels)},
               "label2id": label2id,
               "max_length": MAX_LEN}, f, indent=2)

vocab = tokenizer.get_vocab()
with open(OUTPUT_DIR / "tokenizer" / "vocab.json", "w", encoding="utf-8") as f:
    json.dump({
        "vocab": vocab,
        "unk_token": tokenizer.unk_token,
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "mask_token": getattr(tokenizer, "mask_token", "[MASK]"),
        "do_lower_case": True
    }, f, ensure_ascii=False)

print("✓ Training complete. Saved to", OUTPUT_DIR)
