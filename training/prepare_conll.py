# training/prepare_conll.py
from pathlib import Path
from typing import List, Tuple

def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads a CoNLL-like file where each non-empty line has at least:
    TOKEN ... TAG
    Sentences are separated by blank lines.
    """
    sents_tokens, sents_tags = [], []
    cur_tokens, cur_tags = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if cur_tokens:
                    sents_tokens.append(cur_tokens)
                    sents_tags.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                continue
            parts = line.split()
            token = parts[0]
            tag = parts[-1]
            cur_tokens.append(token)
            cur_tags.append(tag)
    if cur_tokens:
        sents_tokens.append(cur_tokens)
        sents_tags.append(cur_tags)
    return sents_tokens, sents_tags

if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent          # .../training
    DATA_DIR = HERE / "data"                        # .../training/data
    train_path = DATA_DIR / "train.conll"
    dev_path   = DATA_DIR / "dev.conll"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(f"Missing train/dev under {DATA_DIR}")

    tr_toks, tr_tags = read_conll(str(train_path))
    dv_toks, dv_tags = read_conll(str(dev_path))

    if not tr_toks or not dv_toks:
        raise RuntimeError("Empty train/dev after parsing.")
    if len(tr_toks) != len(tr_tags):
        raise RuntimeError(f"Train tokens vs tags mismatch: {len(tr_toks)} vs {len(tr_tags)}")
    if len(dv_toks) != len(dv_tags):
        raise RuntimeError(f"Dev tokens vs tags mismatch: {len(dv_toks)} vs {len(dv_tags)}")

    print(f"✓ CoNLL OK | data dir={DATA_DIR} | train_sents={len(tr_toks)} dev_sents={len(dv_toks)}")
