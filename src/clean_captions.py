from pathlib import Path
import pandas as pd
import re


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "prepared"

CAPTIONS_FILE = RAW_DATA_DIR / "captions.txt"
OUTPUT_FILE = PREPARED_DATA_DIR / "cleaned_captions.csv"


# ============================================================
# 2. CLEAN ONE CAPTION
# ============================================================

def clean_caption(caption):
    """
    Clean a single caption.
    """

    # Convert to string
    caption = str(caption)

    # Convert to lowercase
    caption = caption.lower()

    # Remove punctuation and special characters
    caption = re.sub(r"[^a-z\s]", " ", caption)

    # Remove extra spaces
    caption = re.sub(r"\s+", " ", caption).strip()

    # Add start and end tokens
    caption = "startseq " + caption + " endseq"

    return caption


# ============================================================
# 3. LOAD CAPTIONS
# ============================================================

def load_captions():

    print("\n[1] Loading captions...")

    df = pd.read_csv(CAPTIONS_FILE)

    print(f"Original rows: {len(df):,}")
    print(f"Original images: {df['image'].nunique():,}")

    return df


# ============================================================
# 4. CLEAN CAPTIONS
# ============================================================

def clean_captions(df):

    print("\n[2] Cleaning captions...")

    # Remove missing values
    df = df.dropna(subset=["image", "caption"]).copy()

    # Clean every caption
    df["caption"] = df["caption"].apply(clean_caption)

    # Remove empty captions
    df = df[df["caption"].str.strip() != ""]

    # Remove duplicate image-caption pairs
    df = df.drop_duplicates(subset=["image", "caption"])

    print(f"Cleaned rows: {len(df):,}")
    print(f"Cleaned images: {df['image'].nunique():,}")

    return df


# ============================================================
# 5. VERIFY CLEANED DATA
# ============================================================

def verify_captions(df):

    print("\n[3] Verifying cleaned captions...")

    # Check startseq
    start_ok = df["caption"].str.startswith("startseq ").all()

    # Check endseq
    end_ok = df["caption"].str.endswith(" endseq").all()

    print(f"All captions start with startseq: {start_ok}")
    print(f"All captions end with endseq: {end_ok}")

    # Captions per image
    caption_counts = df.groupby("image").size()

    print("\nCaptions per image:")
    print(caption_counts.value_counts().sort_index())

    # Show examples
    print("\nSample cleaned captions:")

    for caption in df["caption"].head(5):
        print("-", caption)


# ============================================================
# 6. SAVE CLEANED DATA
# ============================================================

def save_cleaned_data(df):

    print("\n[4] Saving cleaned captions...")

    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to: {OUTPUT_FILE}")


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CAPTION CLEANING")
    print("=" * 60)

    df = load_captions()

    df = clean_captions(df)

    verify_captions(df)

    save_cleaned_data(df)

    print("\n" + "=" * 60)
    print("CAPTION CLEANING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()