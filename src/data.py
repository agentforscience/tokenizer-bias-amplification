"""Dataset access. Files are pulled straight from the HF hub (parquet / tsv) into HF_HOME."""
import pandas as pd
from huggingface_hub import hf_hub_download

from langs import FLORES

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
SIB_LABELS = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]
NUM_LABELS = {"xnli": 3, "ner": len(NER_LABELS), "sib": len(SIB_LABELS)}


def _get(repo, path):
    return hf_hub_download(repo, path, repo_type="dataset")


def load_xnli(lang, split):
    """Columns: premise, hypothesis, label (0 entail, 1 neutral, 2 contradiction)."""
    return pd.read_parquet(_get("facebook/xnli", f"{lang}/{split}-00000-of-00001.parquet"))


def load_ner(lang, split):
    """Columns: tokens (list[str]), ner_tags (list[int]) in NER_LABELS order."""
    df = pd.read_parquet(_get("unimelb-nlp/wikiann", f"{lang}/{split}-00000-of-00001.parquet"))
    return df[["tokens", "ner_tags"]]


def load_sib(lang, split):
    """Columns: text, label. Sentences are Flores-200, parallel across languages."""
    df = pd.read_csv(_get("Davlan/sib200", f"data/{FLORES[lang]}/{split}.tsv"), sep="\t", quoting=3)
    df["label"] = df["category"].map(SIB_LABELS.index)
    return df[["index_id", "text", "label"]]


def flores_text(lang):
    """All 1,004 Flores sentences of a language, ordered by sentence id (parallel across langs)."""
    df = pd.concat([load_sib(lang, s) for s in ("train", "dev", "test")])
    return df.sort_values("index_id").reset_index(drop=True)


LOADERS = {"xnli": load_xnli, "ner": load_ner, "sib": load_sib}
DEV_SPLIT = {"xnli": "validation", "ner": "validation", "sib": "dev"}
