"""Download every dataset file and both models once, so parallel training jobs only read the cache."""
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoModel, AutoTokenizer

from data import load_ner, load_sib, load_xnli
from langs import MODELS, WIKI, XNLI_LANGS

jobs = [(load_xnli, "en", "train"), (load_xnli, "en", "validation"), (load_ner, "en", "train"),
        (load_ner, "en", "validation")]
jobs += [(load_xnli, l, "test") for l in XNLI_LANGS]
jobs += [(load_ner, l, "test") for l in WIKI]
jobs += [(load_sib, l, s) for l in WIKI for s in ("train", "dev", "test")]
with ThreadPoolExecutor(8) as ex:
    sizes = list(ex.map(lambda j: len(j[0](j[1], j[2])), jobs))
for (fn, l, s), n in zip(jobs, sizes):
    print(fn.__name__, l, s, n)
for m in MODELS.values():
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m)
    print("cached", m)
