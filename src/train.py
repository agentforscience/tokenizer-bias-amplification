"""Fine-tune a multilingual encoder on English and evaluate zero-shot on every target language.

Usage: python src/train.py --task {xnli,ner,sib} --model {mbert,xlmr} --seed N [--smoke]
Writes results/model_outputs/{task}_{model}_s{seed}.json (+ .npz predictions) and a
checkpoint under CKPT_DIR for the intervention experiment.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from seqeval.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from data import DEV_SPLIT, LOADERS, NER_LABELS, NUM_LABELS
from langs import CKPT_DIR, MODELS, RESULTS, WIKI, XNLI_LANGS

HP = {  # standard recipes; nothing is tuned on target languages
    "xnli": dict(epochs=2, bs=32, lr=2e-5, max_len=128, n_train=100_000),  # fixed random subset of MNLI
    "ner": dict(epochs=5, bs=32, lr=3e-5, max_len=256),
    "sib": dict(epochs=20, bs=16, lr=2e-5, max_len=128),
}
TASK_LANGS = {"xnli": XNLI_LANGS, "ner": WIKI, "sib": WIKI}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, task, df, max_len):
    """Standard tokenization. Returns a list of feature dicts (python lists, unpadded)."""
    if task == "xnli":
        enc = tok(list(df.premise), list(df.hypothesis), truncation=True, max_length=max_len)
    elif task == "sib":
        enc = tok(list(df.text), truncation=True, max_length=max_len)
    else:
        enc = tok([list(t) for t in df.tokens], is_split_into_words=True, truncation=True, max_length=max_len)
    feats = []
    for i in range(len(df)):
        f = {"input_ids": enc["input_ids"][i]}
        if "token_type_ids" in enc:
            f["token_type_ids"] = enc["token_type_ids"][i]
        if task == "ner":
            # label only the first subword of each word; -100 elsewhere
            tags, lab, prev = df.ner_tags.iloc[i], [], None
            for w in enc.word_ids(i):
                lab.append(-100 if w is None or w == prev else int(tags[w]))
                prev = w
            f["labels"] = lab
        else:
            f["labels"] = int(df.label.iloc[i])
        feats.append(f)
    return feats


def collate(feats, pad_id, device):
    n = max(len(f["input_ids"]) for f in feats)
    ids = torch.full((len(feats), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(feats), n), dtype=torch.long)
    batch = {}
    if "token_type_ids" in feats[0]:
        tt = torch.zeros((len(feats), n), dtype=torch.long)
    seq_labels = isinstance(feats[0]["labels"], list)
    lab = torch.full((len(feats), n), -100, dtype=torch.long) if seq_labels else torch.tensor([f["labels"] for f in feats])
    for i, f in enumerate(feats):
        k = len(f["input_ids"])
        ids[i, :k] = torch.tensor(f["input_ids"])
        mask[i, :k] = 1
        if "token_type_ids" in f:
            tt[i, :k] = torch.tensor(f["token_type_ids"])
        if seq_labels:
            lab[i, :k] = torch.tensor(f["labels"])
    batch.update(input_ids=ids, attention_mask=mask, labels=lab)
    if "token_type_ids" in feats[0]:
        batch["token_type_ids"] = tt
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model, feats, pad_id, device, bs=128):
    """Returns per-example predictions in the original order (list of ints / list of int lists)."""
    model.eval()
    order = np.argsort([len(f["input_ids"]) for f in feats])
    out = [None] * len(feats)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = collate([feats[i] for i in idx], pad_id, device)
        labels = b.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(**b).logits.argmax(-1)
        pred, labels = pred.cpu(), labels.cpu()  # one transfer per batch, not per example
        if labels.dim() == 2:
            for j, i in enumerate(idx):
                out[i] = pred[j][labels[j] != -100].tolist()
        else:
            for j, i in enumerate(idx):
                out[i] = int(pred[j])
    return out


def score(task, feats, preds, df):
    """Accuracy (xnli, sib) or seqeval span F1 (ner), in percent."""
    if task != "ner":
        return 100 * float(np.mean(np.array(preds) == df.label.values))
    gold, hyp = [], []
    for p, tags in zip(preds, df.ner_tags):
        # words lost to truncation are predicted as O
        p = list(p) + [0] * (len(tags) - len(p))
        gold.append([NER_LABELS[t] for t in tags])
        hyp.append([NER_LABELS[t] for t in p])
    return 100 * f1_score(gold, hyp)


def load_model(task, name_or_path):
    cls = AutoModelForTokenClassification if task == "ner" else AutoModelForSequenceClassification
    return cls.from_pretrained(name_or_path, num_labels=NUM_LABELS[task])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    a = ap.parse_args()
    hp, device = HP[a.task], "cuda"
    tag = f"{a.task}_{a.model}_s{a.seed}" + ("_smoke" if a.smoke else "")
    out_dir = os.path.join(RESULTS, "model_outputs")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(a.seed)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    load = LOADERS[a.task]
    train_df, dev_df = load("en", "train"), load("en", DEV_SPLIT[a.task])
    langs = TASK_LANGS[a.task]
    if "n_train" in hp:
        # GPUs are shared and slow, so train on a fixed subset (same for every seed and model)
        train_df = train_df.sample(n=hp["n_train"], random_state=0).reset_index(drop=True)
    if a.smoke:
        train_df, langs = train_df.iloc[:2000], langs[:3]
    train, dev = encode(tok, a.task, train_df, hp["max_len"]), encode(tok, a.task, dev_df, hp["max_len"])
    print(f"[{tag}] train={len(train)} dev={len(dev)} tokenized in {time.time()-t0:.0f}s", flush=True)

    ck = os.path.join(CKPT_DIR, tag)
    trained = os.path.exists(os.path.join(ck, "config.json")) and not a.smoke
    model = load_model(a.task, ck if trained else MODELS[a.model]).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=hp["lr"])
    epochs = 1 if a.smoke else hp["epochs"]
    steps = epochs * ((len(train) + hp["bs"] - 1) // hp["bs"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(a.seed)

    best_dev, best_state, dev_hist, step = -1, None, [], 0
    for ep in range(0 if trained else epochs):
        model.train()
        perm = rng.permutation(len(train))
        for s in range(0, len(perm), hp["bs"]):
            b = collate([train[i] for i in perm[s:s + hp["bs"]]], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 1000 == 0:
                print(f"[{tag}] step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        d = score(a.task, dev, predict(model, dev, tok.pad_token_id, device), dev_df)
        dev_hist.append(d)
        print(f"[{tag}] epoch {ep+1} en-dev {d:.2f}", flush=True)
        # SIB's train set is tiny, so keep the best English-dev epoch; other tasks use the last epoch
        if a.task == "sib" and d > best_dev:
            best_dev, best_state = d, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    if not trained:
        assert np.isfinite(loss.item())
        if not a.smoke:
            model.save_pretrained(ck)
            json.dump(dev_hist, open(os.path.join(ck, "dev_history.json"), "w"))
    else:
        dev_hist = json.load(open(os.path.join(ck, "dev_history.json")))
        print(f"[{tag}] reusing checkpoint {ck}", flush=True)

    scores, preds_out = {}, {}
    for lang in langs:
        df = load(lang, "test")
        feats = encode(tok, a.task, df, hp["max_len"])
        preds = predict(model, feats, tok.pad_token_id, device)
        scores[lang] = score(a.task, feats, preds, df)
        if a.task != "ner":
            preds_out[lang] = np.array(preds, dtype=np.int8)
        print(f"[{tag}] test {lang} {scores[lang]:.2f} (n={len(df)})", flush=True)

    res = dict(task=a.task, model=a.model, hf_model=MODELS[a.model], seed=a.seed, hp=hp, dev_history=dev_hist,
               scores=scores, n_train=len(train), runtime_s=time.time() - t0, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    if preds_out:
        np.savez_compressed(os.path.join(out_dir, tag + "_preds.npz"), **preds_out)
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
