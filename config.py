"""Shared settings for Stage 3 and Stage 4. Place this file in your project root
(the folder that contains data/raw and data/prepared)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "data" / "raw" / "Images"
PREP_DIR = ROOT / "data" / "prepared"
FEAT_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
for _d in (FEAT_DIR, MODEL_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = FEAT_DIR / "resnet50_features.pt"
VOCAB_FILE = PREP_DIR / "vocab.json"
BEST_MODEL = MODEL_DIR / "best_model.pt"

# ---- vocabulary ----
MIN_WORD_FREQ = 5      # words rarer than this become <unk>
MAX_LEN = 34           # max words per caption (excluding startseq/endseq)

# ---- model ----
FEATURE_DIM = 2048     # ResNet50 pooled feature size
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.5

# ---- training ----
BATCH_SIZE = 64
EPOCHS = 30
LR = 4e-4
GRAD_CLIP = 5.0
PATIENCE = 5           # early-stopping patience (epochs)
SEED = 42

# ---- flexible CSV loader (handles different column names from Stage 2) ----
import re as _re
import pandas as _pd

_IMG_COLS = ["image_name", "image", "filename", "file_name", "img", "image_id"]
_CAP_COLS = ["cleaned_caption", "clean_caption", "caption_cleaned",
             "processed_caption", "caption", "captions", "text"]


def _clean(text):
    words = _re.sub(r"[^a-z ]", " ", str(text).lower()).split()
    words = [w for w in words if len(w) > 1 or w == "a"]
    return "startseq " + " ".join(words) + " endseq"


def load_split(name):
    """Load train / validation / test CSV and return a DataFrame with the
    columns `image_name` and `cleaned_caption` (startseq ... endseq)."""
    path = PREP_DIR / f"{name}.csv"
    df = _pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    img = next((lower[c] for c in _IMG_COLS if c in lower), None)
    cap = next((lower[c] for c in _CAP_COLS if c in lower), None)
    if img is None or cap is None:
        raise KeyError(f"Cannot find image/caption columns in {path}. "
                       f"Columns present: {list(df.columns)}")
    out = _pd.DataFrame({"image_name": df[img].astype(str).str.strip(),
                         "cleaned_caption": df[cap].astype(str)})
    if not out["cleaned_caption"].str.contains("startseq").all():
        out["cleaned_caption"] = out["cleaned_caption"].map(_clean)
    return out