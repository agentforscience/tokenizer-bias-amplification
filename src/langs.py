"""Language table shared by all scripts.

Each row: wiki code (WikiANN / Wikipedia / CC-100 key), ISO 639-3 candidates for URIEL
(first one found in lang2vec is used), Flores-200 code (SIB-200 config), XNLI flag.

The set is the intersection of: mBERT's pretraining list (datasets/meta/mbert_multilingual.md),
XLM-R's pretraining list (CC-100 languages listed in the XLM-R paper), Flores-200, WikiANN.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/dev/shm/tokbias_hf")
CKPT_DIR = os.environ.get("TOKBIAS_CKPT", "/dev/shm/tokbias_ckpt")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

MODELS = {"mbert": "bert-base-multilingual-cased", "xlmr": "xlm-roberta-base"}

# wiki, uriel candidates, flores
_TABLE = """
en eng eng_Latn
af afr afr_Latn
sq sqi,als als_Latn
ar arb,ara arb_Arab
hy hye hye_Armn
az azj,aze azj_Latn
eu eus eus_Latn
be bel bel_Cyrl
bn ben ben_Beng
bs bos bos_Latn
bg bul bul_Cyrl
my mya mya_Mymr
ca cat cat_Latn
zh cmn,zho zho_Hans
hr hrv hrv_Latn
cs ces ces_Latn
da dan dan_Latn
nl nld nld_Latn
et est,ekk est_Latn
fi fin fin_Latn
fr fra fra_Latn
gl glg glg_Latn
ka kat kat_Geor
de deu deu_Latn
el ell ell_Grek
gu guj guj_Gujr
he heb heb_Hebr
hi hin hin_Deva
hu hun hun_Latn
is isl isl_Latn
id ind ind_Latn
ga gle gle_Latn
it ita ita_Latn
ja jpn jpn_Jpan
jv jav jav_Latn
kn kan kan_Knda
kk kaz kaz_Cyrl
ky kir kir_Cyrl
ko kor kor_Hang
lv lav,lvs lvs_Latn
lt lit lit_Latn
mk mkd mkd_Cyrl
mg plt,mlg plt_Latn
ms zsm,msa zsm_Latn
ml mal mal_Mlym
mr mar mar_Deva
mn khk,mon khk_Cyrl
ne npi,nep npi_Deva
no nor,nob nob_Latn
fa pes,fas pes_Arab
pl pol pol_Latn
pt por por_Latn
pa pan pan_Guru
ro ron ron_Latn
ru rus rus_Cyrl
sr srp srp_Cyrl
sk slk slk_Latn
sl slv slv_Latn
es spa spa_Latn
su sun sun_Latn
sw swh,swa swh_Latn
sv swe swe_Latn
tl tgl tgl_Latn
ta tam tam_Taml
te tel tel_Telu
th tha tha_Thai
tr tur tur_Latn
uk ukr ukr_Cyrl
ur urd urd_Arab
uz uzn,uzb uzn_Latn
vi vie vie_Latn
cy cym cym_Latn
"""

LANGS = []
for line in _TABLE.strip().splitlines():
    w, u, f = line.split()
    LANGS.append({"lang": w, "uriel": u.split(","), "flores": f, "script": f.split("_")[1]})

WIKI = [r["lang"] for r in LANGS]
FLORES = {r["lang"]: r["flores"] for r in LANGS}
XNLI_LANGS = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
# Not whitespace-segmented: word-level fertility is undefined, excluded from primary analysis.
UNSEGMENTED = ["zh", "ja", "th", "my"]
CC100_KEY = {"zh": "zh-Hans"}  # CC-100 file key where it differs from the wiki code
assert len(WIKI) == len(set(WIKI)) == 72 and set(XNLI_LANGS) <= set(WIKI)
