# training/export_tfjs_from_hf.py — convert an existing HF TF model to web/model_mini_int8
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from transformers import TFAutoModelForTokenClassification, AutoConfig

HERE = Path(__file__).resolve().parent          # .../training
WEB_DIR = HERE.parent / "web"                   # .../web
DST = WEB_DIR / "model_mini_int8"
TMP_SM_DIR = HERE / "tmp_savedmodel"
HF_DIR = HERE / "mini_legalbert_ner_tf"         # default to student produced by distill
MAX_LEN = 128

if __name__ == "__main__":
    cfg = AutoConfig.from_pretrained(str(HF_DIR))
    model = TFAutoModelForTokenClassification.from_pretrained(str(HF_DIR), config=cfg, from_pt=False)

    dummy = {
        "input_ids":      tf.constant(np.zeros((1, MAX_LEN), dtype=np.int32)),
        "attention_mask": tf.constant(np.ones((1, MAX_LEN),  dtype=np.int32)),
        "token_type_ids": tf.constant(np.zeros((1, MAX_LEN), dtype=np.int32)),
    }
    _ = model(dummy, training=False)

    if TMP_SM_DIR.exists():
        shutil.rmtree(TMP_SM_DIR)
    model.save(str(TMP_SM_DIR), save_format="tf")
    print("✓ Saved temporary SavedModel to", TMP_SM_DIR)

    DST.mkdir(parents=True, exist_ok=True)
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=str(TMP_SM_DIR),
        output_dir=str(DST),
        signature_def="serving_default",
        saved_model_tags="serve",
        skip_op_check=True,
        strip_debug_ops=True
    )
    print("✓ Exported TF.js model to", DST)
