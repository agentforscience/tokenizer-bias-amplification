# Tokenizer Fertility vs. Cross-Lingual Transfer

Tests whether tokenizer fertility (subwords per word, relative to English) predicts
zero-shot transfer degradation better than typological distance. mBERT and XLM-R are
fine-tuned on English and evaluated on up to 71 languages across XNLI, WikiANN NER and
SIB-200, with fertility, URIEL distances and pretraining size as competing predictors.

Full write-up: [REPORT.md](REPORT.md). Pre-registered plan and deviations: [planning.md](planning.md).

## Key findings
- **The hypothesis is not supported.** Fertility explains 0-27% of between-language
  variance in transfer drop (hypothesised: 40-60%); pooled R² = 0.16 [0.07, 0.28].
- **Syntactic distance beats fertility in all six model-task cells** (R² 0.28-0.68);
  pretraining size beats it clearly in four.
- **The 2.5x threshold does not generalise.** No whitespace-segmented language reaches it
  under XLM-R. Under mBERT six do: 5/6 lose >30 F1 on NER, 1/6 lose >30 on SIB-200.
- **Beyond size and syntax, fertility adds at most 0.05 R²**, significant in one cell.
- **Fragmentation still matters causally**: over-segmenting English at test time costs
  10-24 points at about 1.4x fertility; and within a language, the model that fragments it
  more loses more (r = 0.34-0.61).
- Exploratory: a tokens-per-parallel-sentence ratio reaches R² 0.44-0.51 on SIB-200 only.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv sync --no-install-project   # or: uv pip install -r requirements.txt
export HF_HOME=/path/with/space          # default /dev/shm/tokbias_hf
python src/prefetch.py                   # datasets + models (about 2 GB)
python src/build_features.py             # fertility, URIEL distances, corpus sizes
python src/run_all.py                    # 22 fine-tuning runs; resumable, shares GPUs
python src/intervention.py               # over-segmentation of English (needs checkpoints)
python src/analyze.py                    # statistics, tables, figures (about 2 min, CPU)
python src/posthoc.py                    # exploratory checks
```
One GPU with 8 GB is enough; total about 7 GPU-hours on contended RTX A6000s, less on an
idle one. Checkpoints go to `$TOKBIAS_CKPT` (default `/dev/shm/tokbias_ckpt`, about 19 GB)
and were deleted after the run; `analyze.py` needs only the files in `results/`.

## Files
```
planning.md            plan written before the experiments, with deviation log
REPORT.md              full report
src/langs.py           the 72-language table and paths
src/data.py            dataset loaders (HF hub parquet/tsv)
src/train.py           English fine-tuning + zero-shot evaluation
src/run_all.py         job scheduler for shared GPUs
src/build_features.py  predictors per language
src/intervention.py    over-segmentation experiment
src/analyze.py         planned statistics and figures
src/posthoc.py         exploratory analyses
datasets/meta/         source pages for language lists and corpus sizes
results/               scores, predictions, feature tables, statistics
figures/               fig1-fig4
logs/                  training and analysis logs
```

## Caveats
The pre-gathered literature/resources were missing, no LLM API key was available, XNLI was
trained on a 100k subset because the GPUs were shared, and the venv lives outside the
workspace (symlinked) because `/data` was full. Details in REPORT.md sections 3.6 and 6.
