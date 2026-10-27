"""
STAGE 4: TRAINING THE CAPTION MODEL
Run (after stage3_build.py):  python stage4_train.py
"""
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import config as C
from model import CaptionDecoder, count_params
from vocabulary import Vocabulary


class CaptionDS(Dataset):
    def __init__(self, df, feats, name2idx, vocab):
        self.feats = feats
        self.rows = [(name2idx[n], vocab.encode(c, C.MAX_LEN))
                     for n, c in zip(df["image"], df["caption"])
                     if n in name2idx]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        idx, ids = self.rows[i]
        return self.feats[idx].float(), torch.tensor(ids, dtype=torch.long)


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    caps = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
    return feats, caps


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total, ntok = 0.0, 0
    with torch.set_grad_enabled(train):
        for feats, caps in loader:
            feats, caps = feats.to(device), caps.to(device)
            logits = model(feats, caps[:, :-1])
            targets = caps[:, 1:]
            loss = crit(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            n = (targets != 0).sum().item()
            if train:
                optimizer.zero_grad()
                (loss / n).backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                optimizer.step()
            total += loss.item()
            ntok += n
    return total / ntok


def show_samples(model, vocab, val_df, feats, name2idx, device, k=5):
    print("\nSAMPLE CAPTIONS (validation images, greedy decoding)")
    names = random.sample(sorted(name2idx.keys() & set(val_df["image"])), k)
    x = torch.stack([feats[name2idx[n]].float() for n in names]).to(device)
    preds = model.generate(x, vocab.start_idx, vocab.end_idx, C.MAX_LEN + 1)
    lines = []
    for n, p in zip(names, preds):
        ref = val_df[val_df["image"] == n]["caption"].iloc[0]
        ref = ref.replace("startseq", "").replace("endseq", "").strip()
        lines.append(f"{n}\n  GEN: {vocab.decode(p)}\n  REF: {ref}\n")
    print("\n".join(lines))
    (C.OUT_DIR / "sample_captions.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_history(hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train")
        plt.plot(hist["epoch"], hist["val_loss"], label="validation")
        plt.xlabel("epoch"); plt.ylabel("loss (per token)"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(C.OUT_DIR / "loss_curve.png", dpi=120)
        print(f"Loss curve saved -> {C.OUT_DIR / 'loss_curve.png'}")
    except ImportError:
        print("matplotlib not installed - skipping loss curve (pip install matplotlib)")


def main():
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, f"\nSTAGE 4: TRAINING  (device: {device})\n", "=" * 70, sep="")

    vocab = Vocabulary.load(C.VOCAB_FILE)
    blob = torch.load(C.FEATURE_FILE)
    feats, name2idx = blob["features"], {n: i for i, n in enumerate(blob["names"])}
    train_df = pd.read_csv(C.PREP_DIR / "train.csv")
    val_df = pd.read_csv(C.PREP_DIR / "validation.csv")
    tr, va = CaptionDS(train_df, feats, name2idx, vocab), CaptionDS(val_df, feats, name2idx, vocab)
    tr_ld = DataLoader(tr, C.BATCH_SIZE, shuffle=True, collate_fn=collate)
    va_ld = DataLoader(va, C.BATCH_SIZE, shuffle=False, collate_fn=collate)
    print(f"Train pairs: {len(tr)}   Val pairs: {len(va)}   Vocab: {len(vocab)}")

    model = CaptionDecoder(len(vocab), C.FEATURE_DIM, C.EMBED_DIM, C.HIDDEN_DIM,
                           C.NUM_LAYERS, C.DROPOUT, vocab.pad_idx).to(device)
    print(f"Trainable parameters: {count_params(model):,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best, bad, hist = float("inf"), 0, {"epoch": [], "train_loss": [], "val_loss": []}
    for ep in range(1, C.EPOCHS + 1):
        t0 = time.time()
        trl = run_epoch(model, tr_ld, opt, device, True)
        val = run_epoch(model, va_ld, opt, device, False)
        sched.step(val)
        for k, v in zip(hist, (ep, trl, val)):
            hist[k].append(v)
        flag = ""
        if val < best:
            best, bad, flag = val, 0, "  <- best, saved"
            torch.save({"model_state": model.state_dict(),
                        "config": dict(feature_dim=C.FEATURE_DIM, embed_dim=C.EMBED_DIM,
                                       hidden_dim=C.HIDDEN_DIM, num_layers=C.NUM_LAYERS,
                                       dropout=C.DROPOUT, vocab_size=len(vocab)),
                        "val_loss": val, "epoch": ep}, C.BEST_MODEL)
        else:
            bad += 1
        print(f"Epoch {ep:02d}/{C.EPOCHS}  train {trl:.4f}  val {val:.4f}  "
              f"ppl {np.exp(val):.1f}  lr {opt.param_groups[0]['lr']:.1e}  "
              f"{time.time() - t0:.0f}s{flag}")
        if bad >= C.PATIENCE:
            print(f"Early stopping (no improvement for {C.PATIENCE} epochs).")
            break

    pd.DataFrame(hist).to_csv(C.OUT_DIR / "training_history.csv", index=False)
    plot_history(hist)
    model.load_state_dict(torch.load(C.BEST_MODEL)["model_state"])
    show_samples(model, vocab, val_df, feats, name2idx, device)
    print(f"Best model -> {C.BEST_MODEL}  (val loss {best:.4f})")
    print("\nSTAGE 4 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
