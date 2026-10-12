# Planning: Tokenizer Fertility and Cross-Lingual Transfer Degradation

Written before any model was trained or evaluated. Deviations are logged at the bottom.

## Motivation & Novelty Assessment

### Why This Research Matters
Multilingual encoders are fine-tuned on English and deployed on other languages. If the
transfer loss is driven by how finely the shared subword vocabulary fragments a language,
that is something an engineer can fix (vocabulary allocation, tokenizer choice), whereas
typological distance is not. Knowing which of the two predicts the loss tells practitioners
where to spend effort and which languages are at risk before any labelled data exists.

### Gap in Existing Work
(The pre-gathered literature review was missing from the workspace; this is from my own
knowledge of the literature.) Rust et al. 2021 ("How Good is Your Tokenizer?") show that
mBERT over-segments some languages relative to monolingual tokenizers and that this hurts
monolingual downstream quality. Ács 2019 introduced fertility for mBERT. Lauscher et al.
2020 and Pires et al. 2019 predict zero-shot transfer from typological (URIEL) distance and
pretraining corpus size; Wu & Dredze 2020 show low-resource languages are poorly served.
What is missing is a head-to-head comparison in which fertility, typological distance and
pretraining size are competing predictors of the *same* transfer drops, across several
tasks and two different tokenizers, with a check of whether fertility is anything more than
a proxy for the other two.

### Our Novel Contribution
1. A direct variance-explained comparison (fertility vs. URIEL distances vs. pretraining
   size) over 71 target languages x 3 tasks x 2 models, with bootstrap CIs.
2. A test of the specific numerical claims: R^2 in 40-60%, a 2.5x threshold with >30% drop.
3. Two designs that separate fertility from language identity: a within-language contrast
   (same language, two tokenizers) and a causal intervention (same language, same model,
   artificially finer segmentation of English at test time).

### Experiment Justification
- Exp 1 (zero-shot transfer, 3 tasks x 2 models x multiple seeds): produces the dependent
  variable. Three tasks so the result is not a property of one benchmark; two models so it
  is not a property of one tokenizer.
- Exp 2 (language features): fertility, URIEL distances, pretraining size - the predictors.
- Exp 3 (regression analysis): answers the hypothesis' four sub-claims.
- Exp 4 (within-language tokenizer contrast): language, typology and test data are held
  fixed; only the tokenizer/model differs. Tests whether fertility differences track drop
  differences once typology cannot vary.
- Exp 5 (over-segmentation intervention on English): the only experiment here that
  manipulates fertility. Tells us how much damage fragmentation alone can do.

## Research Question
Does the tokenizer fertility ratio of a target language (relative to English) explain
40-60% of the between-language variance in zero-shot transfer degradation, more than
typological distance does, with a catastrophic regime above 2.5x, and independently of
pretraining data size?

## Hypothesis Decomposition
- H1: R^2 of `drop ~ fertility_ratio` lies in [0.40, 0.60].
- H2: R^2(fertility) > R^2(typological distance).
- H3: languages with fertility ratio > 2.5 have drop > 30; languages below mostly do not.
- H4: the fertility effect survives controlling for pretraining data size (partial
  correlation stays clearly non-zero; no fertility x size interaction needed to explain it).

Variables. DV: `drop = score_en - score_target` in points (accuracy for XNLI/SIB-200, span
F1 for NER), mean over seeds. Also relative drop `100*drop/score_en`. IVs defined below.

## Proposed Methodology

### Models
`bert-base-multilingual-cased` (WordPiece, 119k) and `xlm-roberta-base` (SentencePiece
unigram, 250k). Both are the standard zero-shot transfer baselines; they need fine-tuning
and per-language tokenization, so local models are required. No LLM API is used: no
OpenRouter key is present in the environment, and API LLMs do not give a clean
"train on English only" transfer setting.

### Tasks and data (all English-train, zero-shot test)
| Task | HF dataset | Train | Target languages |
|---|---|---|---|
| NLI | `facebook/xnli` | en MNLI 393k | 14 |
| NER | `unimelb-nlp/wikiann` | en 20k | 71 |
| Topic classification | `Davlan/sib200` (Flores-200 sentences) | en 701 | 71 |

Language set (fixed in advance): every language that is (a) in mBERT's pretraining list,
(b) in XLM-R's pretraining list, (c) in Flores-200/SIB-200, (d) in WikiANN. 72 languages
including English. Restricting to languages seen by both models removes the "unseen
language" confound.

### Predictors
- Fertility (primary): mean subword tokens per whitespace word on the language's 1,004
  Flores-200 sentences (SIB-200 train+dev+test text), divided by the same quantity for
  English. Chinese, Japanese, Thai and Burmese are not whitespace-segmented, so they are
  excluded from the primary analysis (67 targets; 12 for XNLI).
- Fertility (secondary, all 71): parallel token ratio = total tokens on the Flores
  sentences / total tokens on the English translations of the same sentences.
- Robustness: fertility on WikiANN's own pre-tokenized test words.
- Typological distance from English: cosine distance between URIEL (lang2vec) kNN-imputed
  feature vectors: syntactic (primary baseline, the one used most in prior work),
  phonological, inventory, genetic (family vectors), geographic. Also all five jointly.
- Pretraining size: mBERT - log10 Wikipedia article count (Wikimedia API, today's count as
  a proxy for the 2018 dump); XLM-R - log10 CC-100 size from data.statmt.org/cc-100.
- Covariates reported, not in primary tests: non-Latin script, UNK rate, share of words
  split into 2+ pieces.

### Training
Standard recipes, no per-language tuning, model selection on English dev only.
- XNLI: 2 epochs, bs 32, lr 2e-5, max len 128, 3 seeds.
- NER: 5 epochs, bs 32, lr 3e-5, first-subword labelling, seqeval span F1, 3 seeds.
- SIB-200: 20 epochs, bs 16, lr 2e-5, best English-dev epoch, 5 seeds (small train set).

### Baselines
Predictor baselines for fertility: syntactic distance, each other URIEL distance, all URIEL
distances jointly, log pretraining size, non-Latin script indicator. Performance
baselines: majority class / chance for each task, to put drops in context.

### Statistical Analysis Plan
Unit of analysis: language, within each of the 6 model x task cells.
- H1: OLS R^2 and Spearman rho, 95% percentile bootstrap CI over languages (10,000
  resamples). Supported in a cell if the R^2 point estimate is in [0.40, 0.60].
- H2: bootstrap CI of R^2(fertility) - R^2(syntactic). Also against the joint URIEL model
  using adjusted R^2. Supported if the difference CI excludes 0 in most cells.
- H3: among languages with ratio > 2.5, share with absolute drop > 30 points and share
  with relative drop > 30%; same shares below 2.5; Fisher exact test. Report how many
  languages exceed 2.5 at all. Linear vs. hinge-at-2.5 fit as a nonlinearity check.
- H4: partial correlation of fertility with drop given log size; then given log size +
  syntactic distance; OLS with fertility x size interaction. Unique R^2 of fertility in
  the full model (fertility + size + syntactic).
- Pooled: all cells stacked, drop z-scored within cell, bootstrap clustered by language.
- Holm correction across the 6 cells for each family of tests. alpha = 0.05.
- Seed variability: SD across seeds of each language's score.

## Expected Outcomes
Support: fertility R^2 0.4-0.6 in most cells, beating syntactic distance, partial
correlation given size still large, a visible break near 2.5x, and a dose-response in the
intervention. Refutation: fertility R^2 well below 0.4 or not above the baselines; the
effect vanishing once size is controlled; few or no languages beyond 2.5x, or those
languages not failing; the within-language contrast showing no relationship.
My prior: fertility correlates with drop but is heavily entangled with pretraining size
(the vocabulary is learned from the same corpus) and with script.

## Timeline
Setup and metadata 30 min; training code 45 min; training runs ~60 min wall on 4 GPUs;
intervention 20 min; analysis 45 min; report 30 min.

## Potential Challenges
- /data is full (6 GB): venv under ~/.cache (symlinked), HF cache and checkpoints in
  /dev/shm, checkpoints deleted at the end.
- SIB-200 test sets are 204 sentences: per-language accuracy has SE ~3 points. Mitigated
  with 5 seeds; noise ceiling estimated.
- XNLI has only 14 targets: wide CIs, reported as such.
- Few languages may exceed 2.5x: if so H3 is reported as untestable/unsupported at that
  threshold rather than moving the threshold.

## Success Criteria
All six cells trained with sane English scores (XNLI ~81/84, NER ~83, SIB ~85+), every
pre-specified test reported with CIs whichever way it comes out.

## Deviations from plan
Logged as they happened, before looking at the regression results unless stated.

1. Pretraining size for mBERT: the Wikimedia API rate-limited the machine (HTTP 429), so
   instead of article counts I use the compressed text size of each language's 2023-11-01
   Wikipedia dump (HF datasets-server size endpoint for `wikimedia/wikipedia`). Text volume
   is arguably the better proxy anyway (article counts are inflated by bot-written stubs).
2. URIEL: the installed lang2vec (1.1.2) ships feature vectors but not precomputed
   distances, so distances are computed here (cosine on kNN-imputed features; Euclidean on
   the geographic features, whose cosine has almost no spread). Basque has an empty family
   vector (isolate): genetic distance set to 1. Norwegian uses URIEL code `nor` (the
   `nob` entry has no family information).
3. XNLI training set: the GPUs turned out to be shared with other jobs and ran at 2-4
   steps/s, so XNLI is trained on a fixed random 100k subset of MNLI (same subset for all
   seeds and both models) instead of all 393k. Lower absolute English accuracy is expected.
4. WikiANN: 10 of the 71 target languages have only 100 test sentences (my gu jv kn ky mg
   mn ne pa su). They are kept in the primary analysis (the language set was fixed in
   advance); a robustness row restricts to test sets of >= 1000 sentences.
5. Added covariate analysis `fertility | size + syntactic + script` (script was listed as a
   reported covariate; adding it as a control was decided after seeing that 5 of the 6
   mBERT languages above 2.5x are Brahmic/Georgian-script, before any regression was run).
6. No OpenRouter/LLM API experiments: no API key in the environment (see Models).
7. The pre-gathered resources (literature_review.md, resources.md, papers/, datasets/,
   code/) were absent: the resource-finder stage is marked `skipped` in
   .neurico/pipeline_state.json. Datasets and metadata were fetched in this session.
