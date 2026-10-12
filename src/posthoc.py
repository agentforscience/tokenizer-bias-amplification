"""Post hoc checks, run AFTER seeing the planned results (exploratory; not part of planning.md).

1. Is the higher R² of the parallel token ratio due to the metric or to the four added
   unsegmented languages (zh, ja, th, my)?
2. Does the parallel token ratio survive the same controls as the primary fertility?
3. Which languages below the 2.5x threshold still lose >30 points (mBERT NER)?
4. How much input does the intervention lose to max-length truncation at each cap?
Output: results/posthoc.md
"""
import os

import numpy as np
import pandas as pd

import analyze as A
from data import LOADERS
from intervention import CAPS, Segmenter, build
from langs import RESULTS
from train import HP

md = []
_, df = A.load()
rows = []
for t in A.TASKS:
    for m in A.MODELS:
        d67, d71 = A.cell(df, t, m), A.cell(df, t, m, primary=False)
        y = d67["drop"].values
        r = {"cell": f"{t}|{m}", "n_seg": len(d67),
             "R² fert_ratio (segmented)": A.r2(d67[["fert_ratio"]].values, y),
             "R² par_ratio (segmented)": A.r2(d67[["par_ratio"]].values, y),
             "R² par_ratio (all)": A.r2(d71[["par_ratio"]].values, d71["drop"].values),
             "R² log size (all)": A.r2(d71[["log_size"]].values, d71["drop"].values),
             "R² syntactic (all)": A.r2(d71[["dist_syntactic"]].values, d71["drop"].values),
             "r(fert_ratio, par_ratio)": np.corrcoef(d67.fert_ratio, d67.par_ratio)[0, 1]}
        for name, d in (("segmented", d67), ("all", d71)):
            pr, p = A.partial(d, "par_ratio", ["log_size", "dist_syntactic"])
            r[f"partial par|size+syn ({name})"] = f"{pr:.2f} (p={p:.3g})"
            pr, p = A.partial(d, "par_ratio", ["log_size", "dist_syntactic", "non_latin"])
            r[f"partial par|size+syn+script ({name})"] = f"{pr:.2f} (p={p:.3g})"
        rows.append(r)
md += ["## Parallel token ratio: segmented targets vs. all targets (uncorrected p-values)", pd.DataFrame(rows).round(2).to_markdown(index=False)]

d = A.cell(df, "ner", "mbert")
big = d[d["drop"] > 30].sort_values("drop", ascending=False)
md += ["## mBERT NER: every whitespace-segmented target losing >30 F1",
       big[["lang", "script", "fert_ratio", "drop", "log_size", "dist_syntactic", "ner_test_n"]].round(2).to_markdown(index=False),
       f"Non-Latin targets: {int((d[d.non_latin == 1]['drop'] > 30).sum())}/{int((d.non_latin == 1).sum())} lose >30; "
       f"Latin targets: {int((d[d.non_latin == 0]['drop'] > 30).sum())}/{int((d.non_latin == 0).sum())}. "
       f"Non-Latin and fertility <= 2.5: {int(((d.non_latin == 1) & (d.fert_ratio <= 2.5) & (d['drop'] > 30)).sum())}/"
       f"{int(((d.non_latin == 1) & (d.fert_ratio <= 2.5)).sum())}."]

rows = []
for m in A.MODELS:
    seg = Segmenter(m)
    for t in A.TASKS:
        test = LOADERS[t]("en", "test")
        for cap in CAPS:
            feats, tpw, _ = build(seg, t, test, cap, HP[t]["max_len"])
            kept = sum(len(f["input_ids"]) for f in feats) - len(feats) * (2 if t != "xnli" else (3 if m == "mbert" else 4))
            n_words = (test.premise.str.split().str.len() + test.hypothesis.str.split().str.len()).sum() if t == "xnli" else \
                (test.text.str.split().str.len().sum() if t == "sib" else test.tokens.str.len().sum())
            rows.append({"task": t, "model": m, "cap": cap or 0, "share of tokens truncated": 1 - kept / (tpw * n_words)})
md += ["## Intervention: share of subword tokens cut by the max-length limit",
       pd.DataFrame(rows).pivot(index=["task", "model"], columns="cap", values="share of tokens truncated").round(3).to_markdown()]
open(os.path.join(RESULTS, "posthoc.md"), "w").write("\n\n".join(md) + "\n")
print("\n\n".join(md))
