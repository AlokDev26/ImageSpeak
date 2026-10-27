"""
STAGE 5: EVALUATION ON THE TEST SET (BLEU-1..4, greedy vs beam search)
Self-contained: no changes to model.py are needed.

Run:  python stage5_evaluate.py
Quick test on fewer images:  python stage5_evaluate.py --limit 300
"""
import argparse
import random

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import config as C
from model import CaptionDecoder
from vocabulary import Vocabulary

WEIGHTS = {"BLEU-1": (1, 0, 0, 0), "BLEU-2": (.5, .5, 0, 0),
           "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0), "BLEU-4": (.25, .25, .25, .25)}


def words(caption):
    return [w for w in str(caption).split() if w not in ("startseq", "endseq")]


def bleu_scores(refs, hyps):
    sm = SmoothingFunction().method1
    return {k: corpus_bleu(refs, hyps, weights=w, smoothing_function=sm)
            for k, w in WEIGHTS.items()}


@torch.no_grad()
def beam_search(model, feature, start_idx, end_idx, beam_size=3,
                max_len=35, length_penalty=0.7):
    """Beam search for ONE image. feature: tensor (2048,). Returns token ids."""
    h, c = model.init_state(feature.unsqueeze(0))
    beams = [(0.0, [start_idx], h, c)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for score, seq, h, c in beams:
            inp = torch.tensor([[seq[-1]]], device=feature.device)
            out, (h2, c2) = model.lstm(model.embed(inp), (h, c))
            logp = torch.log_softmax(model.fc(out.squeeze(1)), dim=-1).squeeze(0)
            top_lp, top_ix = logp.topk(beam_size)
            for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                candidates.append((score + lp, seq + [ix], h2, c2))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for cand in candidates[:beam_size]:
            if cand[1][-1] == end_idx:
                finished.append(cand)
            else:
                beams.append(cand)
        if not beams:
            break
    if not finished:
        finished = beams
    best = max(finished, key=lambda x: x[0] / (len(x[1]) ** length_penalty))
    return best[1][1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=3, help="beam size")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only N images")
    args = ap.parse_args()

    random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, f"\nSTAGE 5: EVALUATION  (device: {device})\n", "=" * 70, sep="")

    vocab = Vocabulary.load(C.VOCAB_FILE)
    ck = torch.load(C.BEST_MODEL, map_location=device)
    cfg = ck["config"]
    model = CaptionDecoder(cfg["vocab_size"], cfg["feature_dim"], cfg["embed_dim"],
                           cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
                           vocab.pad_idx).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"Loaded model from epoch {ck['epoch']} (val loss {ck['val_loss']:.4f})")

    blob = torch.load(C.FEATURE_FILE, map_location="cpu")
    name2idx = {n: i for i, n in enumerate(blob["names"])}
    feats = blob["features"]

    test = C.load_split("test")
    refs_by_img = {}
    for n, cap in zip(test["image_name"], test["cleaned_caption"]):
        if n in name2idx:
            refs_by_img.setdefault(n, []).append(words(cap))
    names = sorted(refs_by_img)
    if args.limit:
        names = names[:args.limit]
    refs = [refs_by_img[n] for n in names]
    print(f"Test images: {len(names)}\n")

    # ---- greedy decoding (batched) ----
    greedy = []
    for i in range(0, len(names), 128):
        x = torch.stack([feats[name2idx[n]].float() for n in names[i:i + 128]]).to(device)
        for p in model.generate(x, vocab.start_idx, vocab.end_idx, C.MAX_LEN + 1):
            greedy.append(vocab.decode(p).split())
    print("Greedy decoding done.")

    # ---- beam search (one image at a time) ----
    beam = []
    for j, n in enumerate(names):
        f = feats[name2idx[n]].float().to(device)
        p = beam_search(model, f, vocab.start_idx, vocab.end_idx,
                        beam_size=args.beam, max_len=C.MAX_LEN + 1)
        beam.append(vocab.decode(p).split())
        if (j + 1) % 200 == 0:
            print(f"  beam search {j + 1}/{len(names)}")
    print(f"Beam search (size {args.beam}) done.\n")

    g, b = bleu_scores(refs, greedy), bleu_scores(refs, beam)
    lines = [f"{'Metric':<8} {'Greedy':>8} {'Beam-' + str(args.beam):>8}"]
    lines += [f"{k:<8} {g[k]:>8.4f} {b[k]:>8.4f}" for k in WEIGHTS]
    avg_len = sum(len(h) for h in greedy) / len(greedy)
    vocab_used = len({w for h in greedy for w in h})
    lines += ["", f"Average generated length (greedy): {avg_len:.1f} words",
              f"Distinct words used (greedy)      : {vocab_used}"]
    print("\n".join(lines))

    # ---- save outputs ----
    pd.DataFrame({"image_name": names,
                  "greedy": [" ".join(h) for h in greedy],
                  "beam": [" ".join(h) for h in beam],
                  "reference_1": [" ".join(r[0]) for r in refs]}
                 ).to_csv(C.OUT_DIR / "test_predictions.csv", index=False)
    (C.OUT_DIR / "evaluation_results.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nSAMPLE PREDICTIONS")
    for i in random.sample(range(len(names)), min(6, len(names))):
        print(f"{names[i]}\n  GREEDY: {' '.join(greedy[i])}\n  BEAM  : {' '.join(beam[i])}"
              f"\n  REF   : {' '.join(refs[i][0])}\n")

    print(f"Saved -> {C.OUT_DIR / 'evaluation_results.txt'}")
    print(f"Saved -> {C.OUT_DIR / 'test_predictions.csv'}")
    print("\nSTAGE 5 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
