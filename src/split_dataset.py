from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "prepared"

CLEANED_CAPTIONS_FILE = PREPARED_DATA_DIR / "cleaned_captions.csv"

TRAIN_FILE = PREPARED_DATA_DIR / "train.csv"
VALIDATION_FILE = PREPARED_DATA_DIR / "validation.csv"
TEST_FILE = PREPARED_DATA_DIR / "test.csv"


def create_cleaned_splits():
    print("=" * 60)
    print("CREATING FINAL CLEANED DATASET SPLITS")
    print("=" * 60)

    # Load cleaned captions
    df = pd.read_csv(CLEANED_CAPTIONS_FILE)

    print(f"\nCleaned captions loaded: {len(df):,}")
    print(f"Unique images: {df['image'].nunique():,}")

    # Get unique images
    images = df["image"].unique()

    # Reproduce our original split using the same random seed
    from sklearn.model_selection import train_test_split

    train_images, remaining_images = train_test_split(
        images,
        train_size=6000,
        random_state=42
    )

    validation_images, test_images = train_test_split(
        remaining_images,
        train_size=1000,
        random_state=42
    )

    # Create final splits from CLEANED captions
    train_df = df[df["image"].isin(train_images)].copy()
    validation_df = df[df["image"].isin(validation_images)].copy()
    test_df = df[df["image"].isin(test_images)].copy()

    # Save final cleaned splits
    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\nFINAL SPLITS")
    print("-" * 60)

    print(
        f"TRAIN      : {train_df['image'].nunique():,} images | "
        f"{len(train_df):,} captions"
    )

    print(
        f"VALIDATION : {validation_df['image'].nunique():,} images | "
        f"{len(validation_df):,} captions"
    )

    print(
        f"TEST       : {test_df['image'].nunique():,} images | "
        f"{len(test_df):,} captions"
    )

    # Check overlap
    train_set = set(train_df["image"])
    validation_set = set(validation_df["image"])
    test_set = set(test_df["image"])

    print("\nOVERLAP CHECK")
    print("-" * 60)
    print("Train ∩ Validation:", len(train_set & validation_set))
    print("Train ∩ Test      :", len(train_set & test_set))
    print("Validation ∩ Test :", len(validation_set & test_set))

    # Caption coverage
    print("\nCAPTION COVERAGE")
    print("-" * 60)

    for name, split in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]:
        counts = split.groupby("image").size()

        print(
            f"{name}: minimum captions per image = {counts.min()}, "
            f"maximum = {counts.max()}"
        )

    print("\nFiles saved:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)

    print("\n" + "=" * 60)
    print("FINAL CLEANED SPLITS CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    create_cleaned_splits()