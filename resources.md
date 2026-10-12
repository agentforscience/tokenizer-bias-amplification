# Resources

The resource-finder stage was skipped for this run (`.neurico/pipeline_state.json`), so
nothing was pre-gathered. Everything below was fetched during the experiment session.

## Datasets (Hugging Face hub, downloaded by `src/prefetch.py`)
| Dataset | Use | Size used |
|---|---|---|
| `facebook/xnli` | NLI: English train/validation, 15 test sets | 100k train subset, 5,010 test each |
| `unimelb-nlp/wikiann` | NER: English train/validation, 72 test sets | 20k train; 100-10,000 test |
| `Davlan/sib200` | Topic classification and Flores-200 text for fertility | 701/99/204 per language |

## Models
`bert-base-multilingual-cased`, `xlm-roberta-base`.

## Metadata (saved in `datasets/meta/`)
- `mbert_multilingual.md`: mBERT language list (google-research/bert).
- `cc100.html`: CC-100 language list and file sizes (data.statmt.org/cc-100).
- `hf_wikipedia_size.json`: per-language Wikipedia dump sizes, 2023-11-01
  (datasets-server.huggingface.co size endpoint for `wikimedia/wikipedia`).
- URIEL typological features: `lang2vec` 1.1.2 package data.

## Not available
Papers, baseline code repositories, a literature review, and an OpenRouter API key.
