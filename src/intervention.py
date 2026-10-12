"""Causal check: over-segment *English* test inputs and measure the score loss.

Language, model and test items are fixed; only segmentation granularity changes. Every
subword longer than `cap` characters is re-split greedily (longest match in the vocabulary,
at most `cap` characters per piece), which raises fertility without changing the text.
cap=None is the standard tokenization and must reproduce the standard English score.

Output: results/intervention.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from data import LOADERS
from langs import CKPT_DIR, MODELS, RESULTS
from train import HP, load_model, predict, score

CAPS = [None, 6, 4, 3, 2, 1]


class Segmenter:
    def __init__(self, mkey):
        self.tok = AutoTokenizer.from_pretrained(MODELS[mkey])
        self.vocab = self.tok.get_vocab()
        self.bert = mkey == "mbert"
        self.cache = {}

    def _mark(self, body, word_initial_piece, first_chunk):
        """Vocabulary form of a chunk: mBERT marks continuations (##x), XLM-R marks word starts (▁x)."""
        if self.bert:
            return body if (word_initial_piece and first_chunk) else "##" + body
        return "▁" + body if (word_initial_piece and first_chunk) else body

    def pieces(self, word, cap):
        key = (word, cap)
        if key in self.cache:
            return self.cache[key]
        out = []
        for p in self.tok.tokenize(word):
            if self.bert:
                initial, body = not p.startswith("##"), p[2:] if p.startswith("##") else p
            else:
                initial, body = p.startswith("▁"), p.lstrip("▁")
            if cap is None or len(body) <= cap:
                out.append(p)
                continue
            i = 0
            while i < len(body):
                for n in range(min(cap, len(body) - i), 0, -1):
                    cand = self._mark(body[i:i + n], initial, i == 0)
                    if cand in self.vocab:
                        out.append(cand)
                        i += n
                        break
                else:
                    if not self.bert and initial and i == 0 and "▁" in self.vocab:
                        out.append("▁")  # emit the bare word-start marker, retry the chunk unmarked
                        initial = False
                        continue
                    out.append(self.tok.unk_token)
                    i += 1
        self.cache[key] = out
        return out

    def words_to_ids(self, words, cap):
        """Returns (ids, index of the first piece of each word)."""
        ids, starts = [], []
        for w in words:
            p = self.pieces(w, cap)
            if not p:
                continue
            starts.append(len(ids))
            ids.extend(self.tok.convert_tokens_to_ids(p))
        return ids, starts


def build(seg, task, df, cap, max_len):
    """Manual equivalent of train.encode() with a segmentation cap. Also returns token/word counts."""
    tok, feats, n_tok, n_word, n_unk = seg.tok, [], 0, 0, 0
    cls, sep = tok.cls_token_id, tok.sep_token_id
    for i in range(len(df)):
        if task == "xnli":
            (a, _), (b, _) = seg.words_to_ids(df.premise.iloc[i].split(), cap), seg.words_to_ids(df.hypothesis.iloc[i].split(), cap)
            n_tok += len(a) + len(b)
            n_word += len(df.premise.iloc[i].split()) + len(df.hypothesis.iloc[i].split())
            n_unk += sum(t == tok.unk_token_id for t in a + b)
            n_special = 3 if seg.bert else 4
            while len(a) + len(b) > max_len - n_special:  # longest-first truncation
                (a if len(a) > len(b) else b).pop()
            if seg.bert:
                f = {"input_ids": [cls] + a + [sep] + b + [sep], "token_type_ids": [0] * (len(a) + 2) + [1] * (len(b) + 1)}
            else:
                f = {"input_ids": [cls] + a + [sep, sep] + b + [sep]}
            f["labels"] = int(df.label.iloc[i])
        else:
            words = df.text.iloc[i].split() if task == "sib" else list(df.tokens.iloc[i])
            ids, starts = seg.words_to_ids(words, cap)
            n_tok += len(ids)
            n_word += len(words)
            n_unk += sum(t == tok.unk_token_id for t in ids)
            ids = ids[:max_len - 2]
            f = {"input_ids": [cls] + ids + [sep]}
            if seg.bert:
                f["token_type_ids"] = [0] * len(f["input_ids"])
            if task == "sib":
                f["labels"] = int(df.label.iloc[i])
            else:
                lab = [-100] * len(f["input_ids"])
                for s, t in zip(starts, df.ner_tags.iloc[i]):
                    if s < len(ids):
                        lab[s + 1] = int(t)
                f["labels"] = lab
        feats.append(f)
    return feats, n_tok / n_word, n_unk / n_tok


def main():
    rows = []
    for ck in sorted(glob.glob(os.path.join(CKPT_DIR, "*_s*"))):
        tag = os.path.basename(ck)
        task, mkey, seed = tag.split("_")
        std = json.load(open(os.path.join(RESULTS, "model_outputs", tag + ".json")))["scores"]["en"]
        seg, df = Segmenter(mkey), LOADERS[task]("en", "test")
        model = load_model(task, ck).cuda()
        base_tpw = None
        for cap in CAPS:
            feats, tpw, unk = build(seg, task, df, cap, HP[task]["max_len"])
            s = score(task, feats, predict(model, feats, seg.tok.pad_token_id, "cuda"), df)
            if cap is None:
                base_tpw, base = tpw, s
                # manual pipeline must match the standard tokenizer pipeline
                assert abs(s - std) < 0.5, (tag, s, std)
            rows.append(dict(task=task, model=mkey, seed=int(seed[1:]), cap=cap or 0, tok_per_word=tpw,
                             fert_ratio=tpw / base_tpw, unk_rate=unk, score=s, drop=base - s, std_score=std))
            print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "intervention.csv"), index=False)


if __name__ == "__main__":
    main()
