import json
from collections import Counter

PAD, UNK, START, END = "<pad>", "<unk>", "startseq", "endseq"


class Vocabulary:
    def __init__(self):
        self.itos = [PAD, UNK, START, END]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    pad_idx = property(lambda self: self.stoi[PAD])
    unk_idx = property(lambda self: self.stoi[UNK])
    start_idx = property(lambda self: self.stoi[START])
    end_idx = property(lambda self: self.stoi[END])

    def build(self, captions, min_freq=5):
        counter = Counter()
        for cap in captions:
            counter.update(w for w in str(cap).split() if w not in (START, END))
        for word, n in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if n >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return counter

    def encode(self, caption, max_len=None):
        words = [w for w in str(caption).split() if w not in (START, END)]
        if max_len:
            words = words[:max_len]
        ids = [self.stoi.get(w, self.unk_idx) for w in words]
        return [self.start_idx] + ids + [self.end_idx]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.end_idx:
                break
            if i in (self.pad_idx, self.start_idx):
                continue
            words.append(self.itos[i])
        return " ".join(words)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, indent=1)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, encoding="utf-8") as f:
            v.itos = json.load(f)
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v
