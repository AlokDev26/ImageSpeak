"""
=============================================================================
STAGE 2: DATA COLLECTION & PREPARATION
IMAGE CAPTION GENERATOR PROJECT
DATASET: FLICKR8K
=============================================================================

This script performs the following tasks:

1. Verify the Flickr8k dataset structure
2. Load the original captions
3. Clean and preprocess captions
4. Check whether corresponding images exist
5. Remove invalid image-caption pairs
6. Load the official train/validation/test splits
7. Prepare separate CSV files
8. Analyze the caption dataset
9. Create a dataset summary

Expected dataset structure:

image_caption_project/
│
├── data/
│   ├── raw/
│   │   ├── Images/
│   │   ├── Flickr8k.token.txt
│   │   ├── Flickr_8k.trainImages.txt
│   │   ├── Flickr_8k.devImages.txt
│   │   └── Flickr_8k.testImages.txt
│   │
│   └── prepared/
│       ├── cleaned_captions.csv
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── dataset_summary.txt
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import re
import sys
import pandas as pd
from pathlib import Path




# ============================================================================
# 2. PROJECT PATH CONFIGURATION
# ============================================================================

# Find the project root automatically.
#
# Current file:
# image_caption_project/src/data_preparation.py
#
# parent      -> src
# parent.parent -> image_caption_project

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define important directories.

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "prepared"

# Flickr8k image directory.

IMAGES_DIR = RAW_DATA_DIR / "Images"

# Caption file.

CAPTIONS_FILE = RAW_DATA_DIR / "Flickr8k.token.txt"

# Official dataset split files.

TRAIN_SPLIT_FILE = RAW_DATA_DIR / "Flickr_8k.trainImages.txt"

VALIDATION_SPLIT_FILE = RAW_DATA_DIR / "Flickr_8k.devImages.txt"

TEST_SPLIT_FILE = RAW_DATA_DIR / "Flickr_8k.testImages.txt"


# ============================================================================
# 3. VERIFY DATASET STRUCTURE
# ============================================================================

def verify_dataset():
    """
    Verify that all required Flickr8k files and folders exist.

    Returns:
        bool: True if everything exists.
              False if one or more files are missing.
    """

    print("=" * 70)
    print("STEP 1: VERIFYING DATASET STRUCTURE")
    print("=" * 70)

    required_paths = [
        IMAGES_DIR,
        CAPTIONS_FILE,
        TRAIN_SPLIT_FILE,
        VALIDATION_SPLIT_FILE,
        TEST_SPLIT_FILE,
    ]

    all_found = True

    for path in required_paths:

        if path.exists():

            print(f"[OK] FOUND: {path}")

        else:

            print(f"[MISSING] {path}")

            all_found = False

    # Check image count separately.

    if IMAGES_DIR.exists():

        image_files = list(IMAGES_DIR.glob("*.jpg"))

        print(f"\nNumber of JPG images found: {len(image_files)}")

    if not all_found:

        print("\nERROR: Dataset structure is incomplete.")

        print("\nExpected files:")

        print("  data/raw/Images/")
        print("  data/raw/Flickr8k.token.txt")
        print("  data/raw/Flickr_8k.trainImages.txt")
        print("  data/raw/Flickr_8k.devImages.txt")
        print("  data/raw/Flickr_8k.testImages.txt")

        return False

    print("\nDataset structure verified successfully.")

    return True


# ============================================================================
# 4. LOAD FLICKR8K CAPTIONS
# ============================================================================

def load_captions():
    """
    Load captions from Flickr8k.token.txt.

    Original format:

    1000268201_693b08cb0e.jpg#0
    A child in a pink dress is climbing...

    We convert this into:

    image_name
    caption_number
    caption

    Returns:
        pandas.DataFrame
    """

    print("\n" + "=" * 70)
    print("STEP 2: LOADING CAPTIONS")
    print("=" * 70)

    data = []

    # Open caption file using UTF-8 encoding.

    with open(
        CAPTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            # Remove extra whitespace.

            line = line.strip()

            # Ignore completely empty lines.

            if not line:

                continue

            try:

                # The original Flickr8k file separates
                # image-caption ID and caption with a tab.

                image_caption_id, caption = line.split(
                    "\t",
                    1
                )

                # Example:
                #
                # 1000268201_693b08cb0e.jpg#0
                #
                # Split into:
                #
                # image_name = 1000268201_693b08cb0e.jpg
                # caption_number = 0

                image_name, caption_number = image_caption_id.rsplit(
                    "#",
                    1
                )

                data.append(
                    {
                        "image_name": image_name,
                        "caption_number": int(caption_number),
                        "caption": caption,
                    }
                )

            except ValueError:

                print(
                    f"WARNING: Could not parse line {line_number}"
                )

    # Convert list of dictionaries into a DataFrame.

    captions_df = pd.DataFrame(data)

    print(
        f"\nTotal caption records loaded: {len(captions_df)}"
    )

    print(
        f"Unique images found in captions: "
        f"{captions_df['image_name'].nunique()}"
    )

    print("\nFirst five records:")

    print(captions_df.head())

    return captions_df


# ============================================================================
# 5. CLEAN CAPTIONS
# ============================================================================

def clean_caption(caption):
    """
    Clean a single caption.

    Operations:

    1. Convert text to lowercase
    2. Remove leading/trailing spaces
    3. Replace multiple spaces with one space
    4. Remove unwanted characters
    5. Add START and END tokens

    Example:

    Original:
        "A Dog is Running in the Park!!!"

    Cleaned:
        "startseq a dog is running in the park endseq"
    """

    # Convert caption to lowercase.

    caption = caption.lower()

    # Remove leading and trailing whitespace.

    caption = caption.strip()

    # Replace multiple spaces with a single space.

    caption = re.sub(
        r"\s+",
        " ",
        caption
    )

    # Keep alphabetic characters and spaces.
    #
    # This removes most punctuation and numbers.

    caption = re.sub(
        r"[^a-z\s]",
        "",
        caption
    )

    # Remove extra spaces again.

    caption = re.sub(
        r"\s+",
        " ",
        caption
    ).strip()

    # Add special sequence tokens.

    caption = (
        "startseq "
        + caption
        + " endseq"
    )

    return caption


def clean_captions(captions_df):
    """
    Clean the complete captions DataFrame.
    """

    print("\n" + "=" * 70)
    print("STEP 3: CLEANING CAPTIONS")
    print("=" * 70)

    # Remove rows where the caption is missing.

    original_count = len(captions_df)

    captions_df = captions_df.dropna(
        subset=["caption"]
    )

    # Remove duplicate rows.

    captions_df = captions_df.drop_duplicates()

    # Apply caption cleaning function.

    captions_df["cleaned_caption"] = (
        captions_df["caption"].apply(
            clean_caption
        )
    )

    # Remove captions that became empty.

    captions_df = captions_df[
        captions_df["cleaned_caption"].str.len() > 0
    ]

    print(f"Original records: {original_count}")

    print(
        f"Records after cleaning: "
        f"{len(captions_df)}"
    )

    print("\nExample caption cleaning:\n")

    sample = captions_df.iloc[0]

    print(
        "Original caption:\n",
        sample["caption"]
    )

    print(
        "\nCleaned caption:\n",
        sample["cleaned_caption"]
    )

    return captions_df


# ============================================================================
# 6. VERIFY IMAGE FILES
# ============================================================================

def verify_images(captions_df):
    """
    Check whether every image mentioned in the
    captions file actually exists.

    Removes records with missing images.
    """

    print("\n" + "=" * 70)
    print("STEP 4: VERIFYING IMAGE-CAPTION PAIRS")
    print("=" * 70)

    # Create a list of images that physically exist.

    existing_images = set(
        image_file.name
        for image_file in IMAGES_DIR.glob("*.jpg")
    )

    # Check whether every image name exists.

    captions_df["image_exists"] = (
        captions_df["image_name"].isin(
            existing_images
        )
    )

    missing_images = captions_df[
        ~captions_df["image_exists"]
    ]

    print(
        f"Caption records before verification: "
        f"{len(captions_df)}"
    )

    print(
        f"Records with missing images: "
        f"{len(missing_images)}"
    )

    # Keep only valid image-caption pairs.

    captions_df = captions_df[
        captions_df["image_exists"]
    ].copy()

    # Remove temporary column.

    captions_df.drop(
        columns=["image_exists"],
        inplace=True
    )

    print(
        f"Valid image-caption pairs: "
        f"{len(captions_df)}"
    )

    print(
        f"Unique valid images: "
        f"{captions_df['image_name'].nunique()}"
    )

    return captions_df


# ============================================================================
# 7. LOAD OFFICIAL DATASET SPLITS
# ============================================================================

def load_split_file(file_path):
    """
    Read one Flickr8k split file.

    Each line contains an image filename.

    Returns:
        set of image names.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        image_names = {
            line.strip()
            for line in file
            if line.strip()
        }

    return image_names


def load_official_splits():
    """
    Load the original Flickr8k train,
    validation and test image splits.
    """

    print("\n" + "=" * 70)
    print("STEP 5: LOADING OFFICIAL DATA SPLITS")
    print("=" * 70)

    train_images = load_split_file(
        TRAIN_SPLIT_FILE
    )

    validation_images = load_split_file(
        VALIDATION_SPLIT_FILE
    )

    test_images = load_split_file(
        TEST_SPLIT_FILE
    )

    print(
        f"Training images: {len(train_images)}"
    )

    print(
        f"Validation images: {len(validation_images)}"
    )

    print(
        f"Test images: {len(test_images)}"
    )

    # Check for accidental overlap.

    train_validation_overlap = (
        train_images & validation_images
    )

    train_test_overlap = (
        train_images & test_images
    )

    validation_test_overlap = (
        validation_images & test_images
    )

    print("\nChecking split overlap...")

    print(
        f"Train/Validation overlap: "
        f"{len(train_validation_overlap)}"
    )

    print(
        f"Train/Test overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation/Test overlap: "
        f"{len(validation_test_overlap)}"
    )

    return (
        train_images,
        validation_images,
        test_images
    )


# ============================================================================
# 8. CREATE DATASET DATAFRAMES
# ============================================================================

def create_splits(
    captions_df,
    train_images,
    validation_images,
    test_images
):
    """
    Create separate DataFrames for:

    - Training
    - Validation
    - Testing
    """

    print("\n" + "=" * 70)
    print("STEP 6: CREATING DATASET SPLITS")
    print("=" * 70)

    train_df = captions_df[
        captions_df["image_name"].isin(
            train_images
        )
    ].copy()

    validation_df = captions_df[
        captions_df["image_name"].isin(
            validation_images
        )
    ].copy()

    test_df = captions_df[
        captions_df["image_name"].isin(
            test_images
        )
    ].copy()

    # Sort data for consistency.

    train_df = train_df.sort_values(
        ["image_name", "caption_number"]
    ).reset_index(drop=True)

    validation_df = validation_df.sort_values(
        ["image_name", "caption_number"]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["image_name", "caption_number"]
    ).reset_index(drop=True)

    print("\nDataset split summary:")

    print(
        f"Training caption pairs: {len(train_df)}"
    )

    print(
        f"Training unique images: "
        f"{train_df['image_name'].nunique()}"
    )

    print()

    print(
        f"Validation caption pairs: "
        f"{len(validation_df)}"
    )

    print(
        f"Validation unique images: "
        f"{validation_df['image_name'].nunique()}"
    )

    print()

    print(
        f"Test caption pairs: {len(test_df)}"
    )

    print(
        f"Test unique images: "
        f"{test_df['image_name'].nunique()}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================================
# 9. ANALYZE DATASET
# ============================================================================

def analyze_dataset(captions_df):
    """
    Analyze caption statistics.
    """

    print("\n" + "=" * 70)
    print("STEP 7: ANALYZING THE DATASET")
    print("=" * 70)

    # Calculate caption length in words.

    captions_df["caption_length"] = (
        captions_df["cleaned_caption"]
        .str.split()
        .str.len()
    )

    # Calculate statistics.

    minimum_length = (
        captions_df["caption_length"].min()
    )

    maximum_length = (
        captions_df["caption_length"].max()
    )

    average_length = (
        captions_df["caption_length"].mean()
    )

    # Calculate vocabulary.

    all_words = []

    for caption in captions_df["cleaned_caption"]:

        words = caption.split()

        all_words.extend(words)

    vocabulary = set(all_words)

    print(
        f"Total image-caption pairs: "
        f"{len(captions_df)}"
    )

    print(
        f"Unique images: "
        f"{captions_df['image_name'].nunique()}"
    )

    print(
        f"Vocabulary size: "
        f"{len(vocabulary)}"
    )

    print(
        f"Shortest caption: "
        f"{minimum_length} words"
    )

    print(
        f"Longest caption: "
        f"{maximum_length} words"
    )

    print(
        f"Average caption length: "
        f"{average_length:.2f} words"
    )

    return {
        "total_caption_pairs": len(captions_df),
        "unique_images": (
            captions_df["image_name"].nunique()
        ),
        "vocabulary_size": len(vocabulary),
        "minimum_caption_length": minimum_length,
        "maximum_caption_length": maximum_length,
        "average_caption_length": average_length,
    }


# ============================================================================
# 10. SAVE PREPARED DATA
# ============================================================================

def save_prepared_data(
    captions_df,
    train_df,
    validation_df,
    test_df,
    statistics
):
    """
    Save all prepared datasets.
    """

    print("\n" + "=" * 70)
    print("STEP 8: SAVING PREPARED DATA")
    print("=" * 70)

    # Create prepared directory if necessary.

    PREPARED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Select useful columns.

    cleaned_captions_df = captions_df[
        [
            "image_name",
            "caption_number",
            "caption",
            "cleaned_caption",
        ]
    ]

    # Save complete cleaned dataset.

    cleaned_captions_path = (
        PREPARED_DATA_DIR
        / "cleaned_captions.csv"
    )

    cleaned_captions_df.to_csv(
        cleaned_captions_path,
        index=False
    )

    # Save training dataset.

    train_path = (
        PREPARED_DATA_DIR
        / "train.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    # Save validation dataset.

    validation_path = (
        PREPARED_DATA_DIR
        / "validation.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    # Save test dataset.

    test_path = (
        PREPARED_DATA_DIR
        / "test.csv"
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # Create dataset summary file.

    summary_path = (
        PREPARED_DATA_DIR
        / "dataset_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FLICKR8K DATASET SUMMARY\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Total image-caption pairs: "
            f"{statistics['total_caption_pairs']}\n"
        )

        file.write(
            f"Unique images: "
            f"{statistics['unique_images']}\n"
        )

        file.write(
            f"Vocabulary size: "
            f"{statistics['vocabulary_size']}\n"
        )

        file.write(
            f"Shortest caption length: "
            f"{statistics['minimum_caption_length']}\n"
        )

        file.write(
            f"Longest caption length: "
            f"{statistics['maximum_caption_length']}\n"
        )

        file.write(
            f"Average caption length: "
            f"{statistics['average_caption_length']:.2f}\n"
        )

        file.write("\nDATA SPLITS\n")

        file.write(
            "=" * 50 + "\n"
        )

        file.write(
            f"Training captions: {len(train_df)}\n"
        )

        file.write(
            f"Training images: "
            f"{train_df['image_name'].nunique()}\n"
        )

        file.write(
            f"Validation captions: "
            f"{len(validation_df)}\n"
        )

        file.write(
            f"Validation images: "
            f"{validation_df['image_name'].nunique()}\n"
        )

        file.write(
            f"Test captions: {len(test_df)}\n"
        )

        file.write(
            f"Test images: "
            f"{test_df['image_name'].nunique()}\n"
        )

    print("\nFiles saved successfully:")

    print(
        f"[OK] {cleaned_captions_path}"
    )

    print(
        f"[OK] {train_path}"
    )

    print(
        f"[OK] {validation_path}"
    )

    print(
        f"[OK] {test_path}"
    )

    print(
        f"[OK] {summary_path}"
    )


# ============================================================================
# 11. MAIN FUNCTION
# ============================================================================

def main():

    print("\n")

    print("#" * 70)

    print(
        "STAGE 2: FLICKR8K DATA COLLECTION & PREPARATION"
    )

    print("#" * 70)

    # ------------------------------------------------------------------------
    # STEP 1: VERIFY DATASET
    # ------------------------------------------------------------------------

    if not verify_dataset():

        print(
            "\nStage 2 stopped because the dataset "
            "structure is incomplete."
        )

        sys.exit(1)

    # ------------------------------------------------------------------------
    # STEP 2: LOAD CAPTIONS
    # ------------------------------------------------------------------------

    captions_df = load_captions()

    # ------------------------------------------------------------------------
    # STEP 3: CLEAN CAPTIONS
    # ------------------------------------------------------------------------

    captions_df = clean_captions(
        captions_df
    )

    # ------------------------------------------------------------------------
    # STEP 4: VERIFY IMAGES
    # ------------------------------------------------------------------------

    captions_df = verify_images(
        captions_df
    )

    # ------------------------------------------------------------------------
    # STEP 5: LOAD OFFICIAL SPLITS
    # ------------------------------------------------------------------------

    (
        train_images,
        validation_images,
        test_images,
    ) = load_official_splits()

    # ------------------------------------------------------------------------
    # STEP 6: CREATE DATA SPLITS
    # ------------------------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_splits(
        captions_df,
        train_images,
        validation_images,
        test_images,
    )

    # ------------------------------------------------------------------------
    # STEP 7: ANALYZE DATASET
    # ------------------------------------------------------------------------

    statistics = analyze_dataset(
        captions_df
    )

    # ------------------------------------------------------------------------
    # STEP 8: SAVE PREPARED DATA
    # ------------------------------------------------------------------------

    save_prepared_data(
        captions_df,
        train_df,
        validation_df,
        test_df,
        statistics,
    )

    # ------------------------------------------------------------------------
    # FINAL MESSAGE
    # ------------------------------------------------------------------------

    print("\n")

    print("#" * 70)

    print(
        "STAGE 2 COMPLETED SUCCESSFULLY!"
    )

    print("#" * 70)

    print(
        "\nYour prepared dataset is available at:"
    )

    print(
        PREPARED_DATA_DIR
    )

    print(
        "\nNext Stage:"
    )

    print(
        "STAGE 3: MODEL ARCHITECTURE DESIGN"
    )


# ============================================================================
# 12. RUN PROGRAM
# ============================================================================

if __name__ == "__main__":

    main()