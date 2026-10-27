import pandas as pd
import re

files = {
    "Train": "../data/prepared/train.csv",
    "Validation": "../data/prepared/validation.csv",
    "Test": "../data/prepared/test.csv"
}

print("=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

total_images = 0
total_captions = 0
all_captions = []

for name, path in files.items():

    df = pd.read_csv(path)

    images = df["image"].nunique()
    captions = len(df)

    total_images += images
    total_captions += captions

    all_captions.extend(df["caption"].astype(str).tolist())

    counts = df.groupby("image").size()

    print(f"\n{name}:")
    print(f"Images: {images:,}")
    print(f"Captions: {captions:,}")
    print(f"Minimum captions per image: {counts.min()}")
    print(f"Maximum captions per image: {counts.max()}")

# Remove startseq and endseq for word statistics
clean_text = []

for caption in all_captions:
    caption = caption.replace("startseq", "")
    caption = caption.replace("endseq", "")
    clean_text.append(caption)

# Word lengths
lengths = []

for caption in clean_text:
    words = re.findall(r"\b[a-z]+\b", caption)
    lengths.append(len(words))

# Vocabulary
vocabulary = set()

for caption in clean_text:
    words = re.findall(r"\b[a-z]+\b", caption)
    vocabulary.update(words)

start_count = sum(
    caption.startswith("startseq ")
    for caption in all_captions
)

end_count = sum(
    caption.endswith(" endseq")
    for caption in all_captions
)

print("\n" + "=" * 60)
print("OVERALL")
print("=" * 60)

print(f"Total images: {total_images:,}")
print(f"Total captions: {total_captions:,}")
print(f"Vocabulary size: {len(vocabulary):,}")
print(f"Average words per caption: {sum(lengths) / len(lengths):.2f}")
print(f"Minimum words: {min(lengths)}")
print(f"Maximum words: {max(lengths)}")
print(f"Captions starting with startseq: {start_count:,}")
print(f"Captions ending with endseq: {end_count:,}")

print("=" * 60)
print("DATASET ANALYSIS COMPLETE")
print("=" * 60)