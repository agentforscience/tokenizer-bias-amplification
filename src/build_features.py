"""Compute the per-language predictors: tokenizer fertility, URIEL distances, pretraining size.

Output: results/language_features.csv, one row per (language, model).
"""
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data import flores_text, load_ner
from langs import CC100_KEY, LANGS, MODELS, RESULTS, ROOT, UNSEGMENTED, WIKI

warnings.filterwarnings("ignore")
import lang2vec.lang2vec as l2v  # noqa: E402

META = os.path.join(ROOT, "datasets", "meta")


def word_stats(tok, word_lists):
    """Tokens per word, share of words split into 2+ pieces, UNK share, over lists of words."""
    n_tok = n_word = n_split = n_unk = 0
    for s in range(0, len(word_lists), 500):
        chunk = word_lists[s:s + 500]
        enc = tok(chunk, is_split_into_words=True, add_special_tokens=False)
        for i, words in enumerate(chunk):
            counts = np.bincount([w for w in enc.word_ids(i)], minlength=len(words))
            n_tok += counts.sum()
            n_word += len(words)
            n_split += (counts >= 2).sum()
            n_unk += sum(t == tok.unk_token_id for t in enc["input_ids"][i])
    return n_tok / n_word, n_split / n_word, n_unk / n_tok


def fertility_table():
    rows = []
    for mkey, mname in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mname)
        for lang in WIKI:
            sents = list(flores_text(lang).text)
            tpw, split, unk = word_stats(tok, [s.split() for s in sents])
            n_sent_tok = sum(len(x) for x in tok(sents, add_special_tokens=False)["input_ids"])
            ner = [list(t) for t in load_ner(lang, "test").tokens]
            ner_tpw, _, ner_unk = word_stats(tok, ner)
            rows.append(dict(lang=lang, model=mkey, tok_per_word=tpw, split_share=split, unk_rate=unk,
                             flores_tokens=n_sent_tok, ner_tok_per_word=ner_tpw, ner_unk_rate=ner_unk,
                             ner_test_n=len(ner)))
        print("fertility done", mkey, flush=True)
    df = pd.DataFrame(rows)
    en = df[df.lang == "en"].set_index("model")
    df["fert_ratio"] = df.tok_per_word / df.model.map(en.tok_per_word)        # primary
    df["par_ratio"] = df.flores_tokens / df.model.map(en.flores_tokens)        # secondary
    df["ner_fert_ratio"] = df.ner_tok_per_word / df.model.map(en.ner_tok_per_word)
    return df


def uriel_table():
    """Distance from English on URIEL feature sets (cosine; kNN-imputed features where applicable)."""
    sets = {"syntactic": "syntax_knn", "phonological": "phonology_knn", "inventory": "inventory_knn",
            "genetic": "fam", "geographic": "geo"}
    avail = set(l2v.available_uriel_languages())
    code = {r["lang"]: next(c for c in r["uriel"] if c in avail) for r in LANGS}
    rows = {l: {"uriel_code": c} for l, c in code.items()}
    for name, fs in sets.items():
        feats = l2v.get_features(sorted(set(code.values())), fs)
        en = np.array(feats["eng"], dtype=float)
        for l, c in code.items():
            v = np.array(feats[c], dtype=float)
            if name == "geographic":
                # geo features are distances to fixed reference points, so use Euclidean distance
                rows[l]["dist_" + name] = np.linalg.norm(v - en)
            elif not v.any():
                # empty family vector = isolate (Basque): shares no family node with English
                assert name == "genetic" and l == "eu", (name, l)
                rows[l]["dist_" + name] = 1.0
            else:
                rows[l]["dist_" + name] = 1 - v @ en / (np.linalg.norm(v) * np.linalg.norm(en))
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("lang").reset_index()


def size_table():
    """log10 pretraining-corpus size: Wikipedia dump bytes (mBERT), CC-100 bytes (XLM-R)."""
    # Wikipedia: compressed text size of the 2023-11-01 dump per language, from the HF
    # datasets-server size endpoint for wikimedia/wikipedia (the Wikimedia API rate-limited us).
    hf = json.load(open(os.path.join(META, "hf_wikipedia_size.json")))["size"]["configs"]
    wiki = {c["config"].split(".")[1]: c["num_bytes_parquet_files"] for c in hf}
    html = open(os.path.join(META, "cc100.html")).read()
    unit = {"K": 1e3, "M": 1e6, "G": 1e9}
    cc = {c: float(v) * unit[u] for c, v, u in re.findall(r'">([\w-]+)</a> [^<]*\(([\d.]+)([KMG])\)<br>', html)}
    rows = []
    for l in WIKI:
        rows.append(dict(lang=l, model="mbert", wiki_bytes=wiki[l], log_size=np.log10(wiki[l])))
        b = cc[CC100_KEY.get(l, l)]
        rows.append(dict(lang=l, model="xlmr", cc100_bytes=b, log_size=np.log10(b)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fert, uriel, size = fertility_table(), uriel_table(), size_table()
    df = fert.merge(size, on=["lang", "model"]).merge(uriel, on="lang")
    df["script"] = df.lang.map({r["lang"]: r["script"] for r in LANGS})
    df["non_latin"] = (df.script != "Latn").astype(int)
    df["unsegmented"] = df.lang.isin(UNSEGMENTED).astype(int)
    assert len(df) == 2 * len(WIKI) and not df.drop(columns=["wiki_bytes", "cc100_bytes"]).isna().any().any()
    df.to_csv(os.path.join(RESULTS, "language_features.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(df.pivot(index="lang", columns="model", values=["fert_ratio", "par_ratio", "unk_rate", "log_size"]).round(2))
    print(df[df.model == "mbert"].set_index("lang").filter(like="dist_").describe().round(2))
