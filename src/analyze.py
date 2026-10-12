"""Statistical analysis and figures. Reads results/model_outputs/*.json, results/language_features.csv
and results/intervention.csv; writes results/*.csv, results/stats.json, results/tables.md, figures/*.png.

The analysis follows planning.md: per model x task cell, languages are the unit of analysis.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from langs import FIGURES, RESULTS

RNG = np.random.default_rng(42)
B = 10_000
TASKS, MODELS = ["xnli", "ner", "sib"], ["mbert", "xlmr"]
TASK_NAME = {"xnli": "XNLI (accuracy)", "ner": "WikiANN NER (F1)", "sib": "SIB-200 (accuracy)"}
MODEL_NAME = {"mbert": "mBERT", "xlmr": "XLM-R"}
URIEL = ["dist_syntactic", "dist_phonological", "dist_inventory", "dist_genetic", "dist_geographic"]
PREDICTORS = {  # name -> columns
    "fertility": ["fert_ratio"], "syntactic": ["dist_syntactic"], "phonological": ["dist_phonological"],
    "inventory": ["dist_inventory"], "genetic": ["dist_genetic"], "geographic": ["dist_geographic"],
    "URIEL (all 5)": URIEL, "log pretrain size": ["log_size"], "non-Latin script": ["non_latin"],
}
# chart tokens (validated categorical slots 1-2; text and grid stay neutral)
C = {"mbert": "#2a78d6", "xlmr": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
                     "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


# ---------------------------------------------------------------- data
def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "model_outputs", "*_s[0-9].json"))):
        d = json.load(open(f))
        rows += [dict(task=d["task"], model=d["model"], seed=d["seed"], lang=l, score=s) for l, s in d["scores"].items()]
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["task", "model", "lang"]).score.agg(score="mean", seed_sd="std", n_seeds="count").reset_index()
    en = agg[agg.lang == "en"].set_index(["task", "model"]).score
    agg["en_score"] = [en[(t, m)] for t, m in zip(agg.task, agg.model)]
    agg["drop"] = agg.en_score - agg.score
    agg["rel_drop"] = 100 * agg["drop"] / agg.en_score
    feats = pd.read_csv(os.path.join(RESULTS, "language_features.csv"))
    df = agg.merge(feats, on=["lang", "model"], validate="many_to_one")
    df["log_fert"] = np.log(df.fert_ratio)
    return raw, df


def cell(df, task, model, primary=True):
    d = df[(df.task == task) & (df.model == model) & (df.lang != "en")]
    return (d[d.unsegmented == 0] if primary else d).reset_index(drop=True)


# ---------------------------------------------------------------- statistics helpers
def r2(X, y, adjusted=False):
    X = np.column_stack([np.ones(len(y)), X])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    sst = ((y - y.mean()) ** 2).sum()
    if sst == 0:
        return np.nan
    v = 1 - (res ** 2).sum() / sst
    return 1 - (1 - v) * (len(y) - 1) / (len(y) - X.shape[1]) if adjusted else v


def boot(d, cols, fn):
    """Percentile bootstrap over languages. fn(matrix with columns `cols`) -> scalar."""
    M, n = d[cols].values.astype(float), len(d)
    out = np.array([fn(M[RNG.integers(0, n, n)]) for _ in range(B)], dtype=float)
    return np.nanpercentile(out, [2.5, 97.5], axis=0), out


def partial(d, target, controls, y="drop"):
    """Partial correlation of `target` with y given controls, with the OLS t-test p-value."""
    m = sm.OLS(d[y], sm.add_constant(d[[target] + controls])).fit()
    t = m.tvalues[target]
    return float(t / np.sqrt(t ** 2 + m.df_resid)), float(m.pvalues[target])


def fmt_ci(v, ci):
    return f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


# ---------------------------------------------------------------- per-cell analysis
def analyze_cell(d, fert="fert_ratio", y="drop"):
    yv, out = d[y].values, {"n": len(d)}
    out["drop_mean"], out["drop_sd"], out["drop_min"], out["drop_max"] = [float(f(yv)) for f in (np.mean, np.std, np.min, np.max)]
    preds = dict(PREDICTORS, fertility=[fert])
    # H1 / H2: variance explained by each predictor (set), with bootstrap CIs
    for name, cols in preds.items():
        adj = len(cols) > 1
        v = r2(d[cols].values, yv, adj)
        ci, _ = boot(d, cols + [y], lambda M, adj=adj: r2(M[:, :-1], M[:, -1], adj))
        out[f"r2|{name}"] = (float(v), ci.tolist())
    r, p = stats.pearsonr(d[fert], yv)
    rho, prho = stats.spearmanr(d[fert], yv)
    out.update(pearson_r=float(r), pearson_p=float(p), spearman=float(rho), spearman_p=float(prho))
    out["spearman_syntactic"] = float(stats.spearmanr(d.dist_syntactic, yv)[0])
    out["spearman_size"] = float(stats.spearmanr(d.log_size, yv)[0])
    for other in ("syntactic", "URIEL (all 5)", "log pretrain size"):
        cols, adj = preds[other], len(preds[other]) > 1
        ci, bs = boot(d, [fert] + cols + [y], lambda M: r2(M[:, :1], M[:, -1], adj) - r2(M[:, 1:-1], M[:, -1], adj))
        p_two = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        out[f"diff|{other}"] = (float(r2(d[[fert]].values, yv, adj) - r2(d[cols].values, yv, adj)), ci.tolist(), float(min(1, p_two)))
    # H4: does fertility survive controls?
    out["partial|size"] = partial(d, fert, ["log_size"], y)
    out["partial|size+syn"] = partial(d, fert, ["log_size", "dist_syntactic"], y)
    out["partial|size+syn+script"] = partial(d, fert, ["log_size", "dist_syntactic", "non_latin"], y)
    out["partial_syn|fert+size"] = partial(d, "dist_syntactic", [fert, "log_size"], y)
    out["partial_size|fert+syn"] = partial(d, "log_size", [fert, "dist_syntactic"], y)
    full = [fert, "log_size", "dist_syntactic"]
    out["r2_full"] = float(r2(d[full].values, yv))
    for c in full:
        out[f"unique|{c}"] = float(out["r2_full"] - r2(d[[x for x in full if x != c]].values, yv))
    z = d[[fert, "log_size"]].apply(lambda s: s - s.mean())
    z["inter"] = z[fert] * z.log_size
    m = sm.OLS(yv, sm.add_constant(z)).fit()
    out["interaction"] = (float(m.params["inter"]), float(m.pvalues["inter"]))
    out["corr_fert_size"] = float(stats.pearsonr(d[fert], d.log_size)[0])
    out["corr_fert_syn"] = float(stats.pearsonr(d[fert], d.dist_syntactic)[0])
    out["corr_fert_script"] = float(stats.pearsonr(d[fert], d.non_latin)[0])
    return out


def threshold(d, thr=2.5, fert="fert_ratio"):
    """H3: do languages above the fertility threshold fail catastrophically (>30)?"""
    hi, lo = d[d[fert] > thr], d[d[fert] <= thr]
    o = dict(n_above=len(hi), n_below=len(lo), above_langs=" ".join(hi.lang))
    for col, nm in (("drop", "abs"), ("rel_drop", "rel")):
        a, b = int((hi[col] > 30).sum()), int((lo[col] > 30).sum())
        o[f"above_{nm}>30"], o[f"below_{nm}>30"] = a, b
        o[f"fisher_p_{nm}"] = float(stats.fisher_exact([[a, len(hi) - a], [b, len(lo) - b]])[1]) if len(hi) else np.nan
    o["mean_drop_above"], o["mean_drop_below"] = float(hi["drop"].mean()) if len(hi) else np.nan, float(lo["drop"].mean())
    o["mean_rel_above"], o["mean_rel_below"] = float(hi.rel_drop.mean()) if len(hi) else np.nan, float(lo.rel_drop.mean())
    if len(hi) >= 2:  # nonlinearity: does a hinge at the threshold improve on a straight line?
        X = sm.add_constant(pd.DataFrame({"f": d[fert], "hinge": np.clip(d[fert] - thr, 0, None)}))
        m = sm.OLS(d["drop"], X).fit()
        o["hinge_coef"], o["hinge_p"] = float(m.params["hinge"]), float(m.pvalues["hinge"])
    return o


# ---------------------------------------------------------------- figures
def fig_scatter(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for i, m in enumerate(MODELS):
        for j, t in enumerate(TASKS):
            ax, d = axes[i, j], cell(df, t, m)
            ax.scatter(d.fert_ratio, d["drop"], s=26, color=C[m], edgecolor=C["surface"], linewidth=0.8, zorder=3)
            b, a = np.polyfit(d.fert_ratio, d["drop"], 1)
            xs = np.array([d.fert_ratio.min(), d.fert_ratio.max()])
            ax.plot(xs, a + b * xs, color=C["ink2"], lw=1.5, zorder=2)
            ax.axvline(2.5, color=C["ink2"], lw=1, ls=(0, (4, 3)), zorder=1)
            ax.axhline(30, color=C["ink2"], lw=1, ls=(0, (1, 3)), zorder=1)
            # direct labels only for the extremes
            lab = pd.concat([d.nlargest(4, "fert_ratio"), d.nlargest(3, "drop"), d.nsmallest(2, "drop")]).drop_duplicates("lang")
            for _, r in lab.iterrows():
                ax.annotate(r.lang, (r.fert_ratio, r["drop"]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=C["ink2"])
            ax.set_title(f"{MODEL_NAME[m]} · {TASK_NAME[t]}", loc="left")
            ax.text(0.03, 0.95, f"R² = {r2(d[['fert_ratio']].values, d['drop'].values):.2f}, n = {len(d)}",
                    transform=ax.transAxes, va="top", fontsize=9)
            ax.set_xlabel("Fertility ratio (subwords per word, relative to English)")
            ax.set_ylabel("Transfer drop vs. English (points)")
    fig.suptitle("Zero-shot transfer drop against tokenizer fertility. Dashed: the hypothesised 2.5x threshold; dotted: 30-point drop.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIGURES, "fig1_fertility_vs_drop.png"), dpi=160)
    plt.close(fig)


def fig_r2(stats_cells):
    names = ["fertility", "syntactic", "URIEL (all 5)", "log pretrain size", "non-Latin script"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for i, m in enumerate(MODELS):
        for j, t in enumerate(TASKS):
            ax, s = axes[i, j], stats_cells[f"{t}|{m}"]
            v = np.array([s[f"r2|{n}"][0] for n in names])
            ci = np.array([s[f"r2|{n}"][1] for n in names])
            ypos = np.arange(len(names))[::-1]
            ax.barh(ypos, v, height=0.55, color=C[m], zorder=3)
            ax.hlines(ypos, ci[:, 0], ci[:, 1], color=C["ink"], lw=1.2, zorder=4)
            ax.axvspan(0.4, 0.6, color=C["grid"], alpha=0.6, zorder=0)
            ax.set_yticks(ypos, names if j == 0 else [""] * len(names))
            ax.set_xlim(-0.15, 1)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{MODEL_NAME[m]} · {TASK_NAME[t]} (n={s['n']})", loc="left")
            if i == 1:
                ax.set_xlabel("R² of transfer drop (adjusted R² for the 5-distance model)")
    fig.suptitle("Variance in transfer drop explained by each predictor alone. Lines: 95% bootstrap CI; grey band: hypothesised 40-60%.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGURES, "fig2_r2_by_predictor.png"), dpi=160)
    plt.close(fig)


def fig_contrast(con):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, t in zip(axes, TASKS):
        d = con[con.task == t]
        ax.scatter(d.d_fert, d.d_drop, s=26, color=C["ink2"], edgecolor=C["surface"], linewidth=0.8, zorder=3)
        ax.axhline(0, color=C["ink2"], lw=0.8)
        for _, r in pd.concat([d.nlargest(4, "d_fert"), d.nlargest(2, "d_drop"), d.nsmallest(2, "d_drop")]).drop_duplicates("lang").iterrows():
            ax.annotate(r.lang, (r.d_fert, r.d_drop), xytext=(4, 3), textcoords="offset points", fontsize=8, color=C["ink2"])
        r, _ = stats.pearsonr(d.d_fert, d.d_drop)
        ax.set_title(f"{TASK_NAME[t]}", loc="left")
        ax.text(0.03, 0.95, f"r = {r:.2f}, n = {len(d)}", transform=ax.transAxes, va="top")
        ax.set_xlabel("Fertility ratio: mBERT minus XLM-R")
        ax.set_ylabel("Transfer drop: mBERT minus XLM-R (points)")
    fig.suptitle("Same language, two tokenizers: does the model that fragments a language more also lose more on it?", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGURES, "fig3_within_language_contrast.png"), dpi=160)
    plt.close(fig)


def fig_intervention(df, iv):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, t in zip(axes, TASKS):
        for m in MODELS:
            d = cell(df, t, m)
            ax.scatter(d.fert_ratio, d["drop"], s=16, facecolor="none", edgecolor=C[m], linewidth=0.9, alpha=0.7, zorder=2,
                       label=f"{MODEL_NAME[m]}: other languages (observed)")
            g = iv[(iv.task == t) & (iv.model == m)]
            ax.errorbar(g.fert_ratio, g["drop"], yerr=g.drop_sd, color=C[m], lw=2, marker="o", ms=5, zorder=3,
                        label=f"{MODEL_NAME[m]}: English, over-segmented (intervention)")
            last = g.iloc[-1]
            ax.annotate(MODEL_NAME[m], (last.fert_ratio, last["drop"]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=C["ink2"])
        ax.set_title(TASK_NAME[t], loc="left")
        ax.set_xlabel("Fertility ratio relative to standard English tokenization")
        ax.set_ylabel("Drop vs. standard English score (points)")
        ax.set_xlim(0.8, 4.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, loc="lower center", ncol=4)
    fig.suptitle("Forcing finer segmentation of English at test time (lines) vs. the cross-lingual pattern (hollow points)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(FIGURES, "fig4_intervention.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(FIGURES, exist_ok=True)
    raw, df = load()
    df.to_csv(os.path.join(RESULTS, "transfer_results.csv"), index=False)
    S, md = {"cells": {}, "threshold": {}, "robust": {}}, []
    cells = [(t, m) for t in TASKS for m in MODELS if len(cell(df, t, m))]

    # --- descriptive
    desc = []
    for t, m in cells:
        d, a = cell(df, t, m), cell(df, t, m, primary=False)
        desc.append(dict(task=t, model=m, seeds=int(a.n_seeds.max()), en_score=a.en_score.iloc[0], n_targets=len(a),
                         mean_target=a.score.mean(), mean_drop=a["drop"].mean(), sd_drop=a["drop"].std(), min_drop=a["drop"].min(),
                         max_drop=a["drop"].max(), worst=a.loc[a["drop"].idxmax(), "lang"], mean_seed_sd=a.seed_sd.mean(),
                         fert_min=d.fert_ratio.min(), fert_median=d.fert_ratio.median(), fert_max=d.fert_ratio.max()))
    desc = pd.DataFrame(desc)
    md += ["## Descriptive (all targets; fertility columns on whitespace-segmented targets)", desc.round(2).to_markdown(index=False)]

    # --- primary per-cell analysis
    for t, m in cells:
        d = cell(df, t, m)
        S["cells"][f"{t}|{m}"] = analyze_cell(d)
        S["threshold"][f"{t}|{m}"] = threshold(d)
        print("analyzed", t, m, flush=True)
    keys = list(S["cells"])
    for fam in ("pearson_p", "spearman_p"):
        adj = multipletests([S["cells"][k][fam] for k in keys], method="holm")[1]
        for k, a in zip(keys, adj):
            S["cells"][k][fam + "_holm"] = float(a)
    for fam in ("partial|size", "partial|size+syn", "partial|size+syn+script"):
        adj = multipletests([S["cells"][k][fam][1] for k in keys], method="holm")[1]
        for k, a in zip(keys, adj):
            S["cells"][k][fam + "|holm"] = float(a)
    adj = multipletests([S["cells"][k]["diff|syntactic"][2] for k in keys], method="holm")[1]
    for k, a in zip(keys, adj):
        S["cells"][k]["diff|syntactic|holm"] = float(a)

    rows = []
    for k in keys:
        s = S["cells"][k]
        rows.append({"cell": k, "n": s["n"], **{n: fmt_ci(*s[f"r2|{n}"]) for n in PREDICTORS}})
    md += ["## H1/H2: R² of drop by single predictor (95% bootstrap CI); 'URIEL (all 5)' is adjusted R²", pd.DataFrame(rows).to_markdown(index=False)]
    rows = []
    for k in keys:
        s = S["cells"][k]
        rows.append({"cell": k, "pearson r": round(s["pearson_r"], 2), "p (Holm)": f"{s['pearson_p_holm']:.3g}",
                     "spearman fert": round(s["spearman"], 2), "p (Holm) ": f"{s['spearman_p_holm']:.3g}",
                     "spearman syntactic": round(s["spearman_syntactic"], 2), "spearman log size": round(s["spearman_size"], 2),
                     "R²(fert)-R²(syn)": fmt_ci(*s["diff|syntactic"][:2]), "p boot (Holm)": f"{s['diff|syntactic|holm']:.3g}",
                     "adjR²(fert)-adjR²(URIEL5)": fmt_ci(*s["diff|URIEL (all 5)"][:2]),
                     "R²(fert)-R²(size)": fmt_ci(*s["diff|log pretrain size"][:2])})
    md += ["## H1/H2: correlations and R² differences", pd.DataFrame(rows).to_markdown(index=False)]
    rows = []
    for k in keys:
        s = S["cells"][k]
        rows.append({"cell": k, "r(fert,size)": round(s["corr_fert_size"], 2), "r(fert,syn)": round(s["corr_fert_syn"], 2),
                     "r(fert,non-Latin)": round(s["corr_fert_script"], 2),
                     "partial r fert|size": f"{s['partial|size'][0]:.2f} (p_holm={s['partial|size|holm']:.3g})",
                     "fert|size+syn": f"{s['partial|size+syn'][0]:.2f} (p_holm={s['partial|size+syn|holm']:.3g})",
                     "fert|size+syn+script": f"{s['partial|size+syn+script'][0]:.2f} (p_holm={s['partial|size+syn+script|holm']:.3g})",
                     "syn|fert+size": f"{s['partial_syn|fert+size'][0]:.2f} (p={s['partial_syn|fert+size'][1]:.3g})",
                     "size|fert+syn": f"{s['partial_size|fert+syn'][0]:.2f} (p={s['partial_size|fert+syn'][1]:.3g})",
                     "R² full": round(s["r2_full"], 2), "unique fert": round(s["unique|fert_ratio"], 3),
                     "unique size": round(s["unique|log_size"], 3), "unique syn": round(s["unique|dist_syntactic"], 3),
                     "fert x size (p)": f"{s['interaction'][0]:.2f} ({s['interaction'][1]:.3g})"})
    md += ["## H4: fertility vs. pretraining size and syntactic distance (full model = fert + log size + syntactic)", pd.DataFrame(rows).to_markdown(index=False)]
    md += ["## H3: 2.5x threshold (primary fertility, whitespace-segmented targets)",
           pd.DataFrame([{"cell": k, **v} for k, v in S["threshold"].items()]).round(3).to_markdown(index=False)]
    # the same threshold on the parallel token ratio over all targets
    md += ["## H3 on the parallel token ratio (all targets, incl. zh/ja/th/my)",
           pd.DataFrame([{"cell": f"{t}|{m}", **threshold(cell(df, t, m, primary=False), fert="par_ratio")} for t, m in cells]).round(3).to_markdown(index=False)]

    # --- pooled across cells, cluster bootstrap by language
    pool = pd.concat([cell(df, t, m).assign(cellid=f"{t}|{m}") for t, m in cells])
    zc = ["drop", "fert_ratio", "log_size", "dist_syntactic"]
    pool[[c + "_z" for c in zc]] = pool.groupby("cellid")[zc].transform(lambda s: (s - s.mean()) / s.std())
    langs = pool.lang.unique()
    by_lang = {l: g for l, g in pool.groupby("lang")}

    def pooled_fit(p):
        o = [r2(p[["fert_ratio_z"]].values, p.drop_z.values), r2(p[["dist_syntactic_z"]].values, p.drop_z.values),
             r2(p[["log_size_z"]].values, p.drop_z.values)]
        X = np.column_stack([np.ones(len(p)), p[["fert_ratio_z", "log_size_z", "dist_syntactic_z"]].values])
        return o + list(np.linalg.lstsq(X, p.drop_z.values, rcond=None)[0][1:])
    est = pooled_fit(pool)
    bs = np.array([pooled_fit(pd.concat([by_lang[l] for l in RNG.choice(langs, len(langs))])) for _ in range(2000)])
    ci = np.percentile(bs, [2.5, 97.5], axis=0).T
    names = ["R² fertility", "R² syntactic", "R² log size", "beta fertility (joint)", "beta log size (joint)", "beta syntactic (joint)"]
    S["pooled"] = {n: (float(e), c.tolist()) for n, e, c in zip(names, est, ci)}
    d_fs = bs[:, 0] - bs[:, 1]
    S["pooled"]["R² fert - R² syn"] = (float(est[0] - est[1]), np.percentile(d_fs, [2.5, 97.5]).tolist())
    d_fz = bs[:, 0] - bs[:, 2]
    S["pooled"]["R² fert - R² size"] = (float(est[0] - est[2]), np.percentile(d_fz, [2.5, 97.5]).tolist())
    md += [f"## Pooled over all cells (z-scored within cell, n={len(pool)} rows, {len(langs)} languages, cluster bootstrap by language)",
           pd.DataFrame([{"quantity": k, "estimate [95% CI]": fmt_ci(*v)} for k, v in S["pooled"].items()]).to_markdown(index=False)]

    # --- within-language tokenizer contrast
    feats_z = df.copy()
    feats_z["size_z"] = feats_z.groupby(["task", "model"]).log_size.transform(lambda s: (s - s.mean()) / s.std())
    con_rows, con_all = [], []
    for t in TASKS:
        a, b = [feats_z[(feats_z.task == t) & (feats_z.model == m) & (feats_z.lang != "en") & (feats_z.unsegmented == 0)].set_index("lang") for m in MODELS]
        if not len(a) or not len(b):
            continue
        c = pd.DataFrame({"task": t, "d_fert": a.fert_ratio - b.fert_ratio, "d_drop": a["drop"] - b["drop"], "d_size": a.size_z - b.size_z}).reset_index()
        con_all.append(c)
        r, p = stats.pearsonr(c.d_fert, c.d_drop)
        rho, prho = stats.spearmanr(c.d_fert, c.d_drop)
        ci, _ = boot(c, ["d_fert", "d_drop"], lambda M: np.corrcoef(M[:, 0], M[:, 1])[0, 1])
        pr, pp = partial(c, "d_fert", ["d_size"], y="d_drop")
        con_rows.append({"task": t, "n": len(c), "mean d_fert": c.d_fert.mean(), "mean d_drop": c.d_drop.mean(), "pearson r [CI]": fmt_ci(r, ci),
                         "p": p, "spearman": rho, "spearman p": prho, "partial r | d_size": pr, "partial p": pp})
    con = pd.concat(con_all)
    con.to_csv(os.path.join(RESULTS, "within_language_contrast.csv"), index=False)
    S["contrast"] = con_rows
    md += ["## Within-language contrast (mBERT minus XLM-R, same language and test set)", pd.DataFrame(con_rows).round(3).to_markdown(index=False)]

    # --- robustness of H1 to analysis choices
    variants = {
        "primary (fert_ratio, abs drop)": dict(),
        "relative drop (%)": dict(y="rel_drop"),
        "log fertility": dict(fert="log_fert"),
        "parallel token ratio, all targets": dict(fert="par_ratio", primary=False),
        "NER: fertility on WikiANN words": dict(fert="ner_fert_ratio", only="ner"),
        "NER: test sets >= 1000 sentences": dict(only="ner", min_n=1000),
        "Latin-script targets only": dict(script=0),
        "non-Latin targets only": dict(script=1),
    }
    rows = []
    for name, v in variants.items():
        row = {"variant": name}
        for t, m in cells:
            if v.get("only") and t != v["only"]:
                row[f"{t}|{m}"] = ""
                continue
            d = cell(df, t, m, v.get("primary", True))
            if "min_n" in v:
                d = d[d.ner_test_n >= v["min_n"]]
            if "script" in v:
                d = d[d.non_latin == v["script"]]
            if len(d) < 6:
                row[f"{t}|{m}"] = f"n={len(d)}"
                continue
            f, y = v.get("fert", "fert_ratio"), v.get("y", "drop")
            val = r2(d[[f]].values, d[y].values)
            ci, _ = boot(d, [f, y], lambda M: r2(M[:, :1], M[:, 1]))
            syn = r2(d[["dist_syntactic"]].values, d[y].values)
            row[f"{t}|{m}"] = f"{fmt_ci(val, ci)}; syn {syn:.2f}; n={len(d)}"
            S["robust"][f"{name}|{t}|{m}"] = (float(val), ci.tolist(), float(syn), len(d))
        rows.append(row)
    md += ["## Robustness: R²(fertility) [CI]; R²(syntactic); n", pd.DataFrame(rows).to_markdown(index=False)]

    # --- measurement noise in the drops (binomial sampling of the test set, accuracy tasks)
    rows = []
    for t, m in cells:
        d = cell(df, t, m)
        if t == "ner":
            continue
        n_test = 5010 if t == "xnli" else 204
        p_t, p_e = d.score / 100, d.en_score / 100
        noise = (1e4 * (p_t * (1 - p_t) + p_e * (1 - p_e)) / n_test).mean()
        rows.append({"cell": f"{t}|{m}", "n_test": n_test, "var(drop)": d["drop"].var(), "sampling var": noise,
                     "reliability": 1 - noise / d["drop"].var(), "mean seed SD": d.seed_sd.mean()})
    md += ["## Noise ceiling: share of between-language drop variance not due to test-set sampling", pd.DataFrame(rows).round(3).to_markdown(index=False)]

    # --- error analysis: where does the fertility model miss?
    rows = []
    for t, m in cells:
        d = cell(df, t, m).copy()
        b, a = np.polyfit(d.fert_ratio, d["drop"], 1)
        d["resid"] = d["drop"] - (a + b * d.fert_ratio)
        rows.append({"cell": f"{t}|{m}", "slope (pts per 1.0 fertility)": round(b, 1),
                     "worse than fertility predicts": ", ".join(f"{r.lang} (+{r.resid:.0f})" for r in d.nlargest(5, "resid").itertuples()),
                     "better than fertility predicts": ", ".join(f"{r.lang} ({r.resid:.0f})" for r in d.nsmallest(5, "resid").itertuples()),
                     "mean resid Latin": round(d[d.non_latin == 0].resid.mean(), 1), "mean resid non-Latin": round(d[d.non_latin == 1].resid.mean(), 1)})
    md += ["## Error analysis: residuals of drop ~ fertility", pd.DataFrame(rows).to_markdown(index=False)]

    # --- intervention
    ipath = os.path.join(RESULTS, "intervention.csv")
    if os.path.exists(ipath):
        iv = pd.read_csv(ipath).groupby(["task", "model", "cap"]).agg(fert_ratio=("fert_ratio", "mean"), drop=("drop", "mean"),
                                                                      drop_sd=("drop", "std"), score=("score", "mean"),
                                                                      unk_rate=("unk_rate", "mean"), seeds=("seed", "count")).reset_index()
        iv = iv.sort_values(["task", "model", "fert_ratio"]).fillna({"drop_sd": 0})
        # cross-lingual drop predicted at the same fertility by the cell's regression line
        pred = []
        for r in iv.itertuples():
            d = cell(df, r.task, r.model)
            b, a = np.polyfit(d.fert_ratio, d["drop"], 1)
            pred.append(a + b * r.fert_ratio)
        iv["crosslingual_fit_at_same_fertility"] = pred
        iv.to_csv(os.path.join(RESULTS, "intervention_summary.csv"), index=False)
        md += ["## Intervention: over-segmenting English at test time (cap = max characters per subword, 0 = standard)", iv.round(2).to_markdown(index=False)]
        fig_intervention(df, iv)

    fig_scatter(df)
    fig_r2(S["cells"])
    fig_contrast(con)
    json.dump(S, open(os.path.join(RESULTS, "stats.json"), "w"), indent=1)
    open(os.path.join(RESULTS, "tables.md"), "w").write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
