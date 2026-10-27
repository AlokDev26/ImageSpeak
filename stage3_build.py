"""
STAGE 3: VOCABULARY + CNN FEATURE EXTRACTION + ARCHITECTURE CHECK
Run:  python stage3_build.py
"""
import sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config as C
from model import EncoderCNN, CaptionDecoder, count_params
from vocabulary import Vocabulary


class ImageDS(Dataset):
    def __init__(self, names, transform):
        self.names, self.transform = names, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(C.IMG_DIR / self.names[i]).convert("RGB")
        return self.transform(img)


def step1_vocab():
    print("=" * 70, "\nSTEP 1: BUILDING VOCABULARY (from train.csv only)\n", "=" * 70, sep="")
    train = pd.read_csv(C.PREP_DIR / "train.csv")
    vocab = Vocabulary()
    counter = vocab.build(train["caption"], C.MIN_WORD_FREQ)
    vocab.save(C.VOCAB_FILE)
    lens = train["caption"].str.split().str.len() - 2
    print(f"Distinct words in train        : {len(counter)}")
    print(f"Vocabulary size (freq >= {C.MIN_WORD_FREQ}) : {len(vocab)}")
    print(f"Caption length mean/max        : {lens.mean():.1f} / {lens.max()}")
    print(f"Saved -> {C.VOCAB_FILE}\n")
    return vocab


def step2_features(device):
    print("=" * 70, "\nSTEP 2: EXTRACTING CNN FEATURES (ResNet50, frozen)\n", "=" * 70, sep="")
    if C.FEATURE_FILE.exists():
        print(f"Features already exist at {C.FEATURE_FILE} - skipping.\n")
        return
    names = sorted(set(pd.concat([pd.read_csv(C.PREP_DIR / f"{s}.csv")["image"]
                                  for s in ("train", "validation", "test")])))
    print(f"Unique images: {len(names)}   device: {device}")
    enc = EncoderCNN().to(device)
    loader = DataLoader(ImageDS(names, enc.transform), batch_size=64,
                        num_workers=2, shuffle=False)
    feats = []
    for i, batch in enumerate(loader):
        feats.append(enc(batch.to(device)).cpu().half())
        if i % 10 == 0:
            print(f"  batch {i + 1}/{len(loader)}")
    feats = torch.cat(feats)
    torch.save({"names": names, "features": feats}, C.FEATURE_FILE)
    print(f"Feature matrix: {tuple(feats.shape)}  saved -> {C.FEATURE_FILE}\n")


def step3_check(vocab):
    print("=" * 70, "\nSTEP 3: ARCHITECTURE SANITY CHECK\n", "=" * 70, sep="")
    dec = CaptionDecoder(len(vocab), C.FEATURE_DIM, C.EMBED_DIM, C.HIDDEN_DIM,
                         C.NUM_LAYERS, C.DROPOUT, vocab.pad_idx)
    feats = torch.randn(4, C.FEATURE_DIM)
    caps = torch.randint(4, len(vocab), (4, 12))
    logits = dec(feats, caps)
    assert logits.shape == (4, 12, len(vocab))
    gen = dec.generate(feats, vocab.start_idx, vocab.end_idx, 10)
    print(f"Decoder trainable parameters : {count_params(dec):,}")
    print(f"Forward pass output shape    : {tuple(logits.shape)}  (batch, time, vocab)")
    print(f"Greedy generate works        : {len(gen)} sequences\n")
    print("IMAGE -> ResNet50 (2048) -> init LSTM state -> word-by-word caption")
    print("\nSTAGE 3 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = step1_vocab()
    step2_features(dev)
    step3_check(v)
