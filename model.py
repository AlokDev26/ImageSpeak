"""STAGE 3 - Architecture: frozen CNN encoder (ResNet50) + LSTM decoder."""
import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """Pretrained ResNet50 with the classifier removed. Output: (B, 2048).
    Frozen: we only use it as a feature extractor."""

    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        self.body = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.transform = weights.transforms()  # resize / crop / normalize (PIL -> tensor)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, images):
        return self.body(images).flatten(1)


class CaptionDecoder(nn.Module):
    """Image feature initialises the LSTM state; the LSTM then predicts the
    caption one word at a time (teacher forcing during training)."""

    def __init__(self, vocab_size, feature_dim=2048, embed_dim=256,
                 hidden_dim=512, num_layers=1, dropout=0.5, pad_idx=0):
        super().__init__()
        self.num_layers, self.hidden_dim = num_layers, hidden_dim
        self.feat_drop = nn.Dropout(dropout)
        self.init_h = nn.Linear(feature_dim, num_layers * hidden_dim)
        self.init_c = nn.Linear(feature_dim, num_layers * hidden_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, features):
        f = self.feat_drop(features)
        B = f.size(0)
        h = torch.tanh(self.init_h(f)).view(B, self.num_layers, self.hidden_dim)
        c = torch.tanh(self.init_c(f)).view(B, self.num_layers, self.hidden_dim)
        return h.permute(1, 0, 2).contiguous(), c.permute(1, 0, 2).contiguous()

    def forward(self, features, captions_in):
        """features (B, 2048), captions_in (B, T) -> logits (B, T, V)"""
        state = self.init_state(features)
        out, _ = self.lstm(self.drop(self.embed(captions_in)), state)
        return self.fc(self.drop(out))

    @torch.no_grad()
    def generate(self, features, start_idx, end_idx, max_len=35):
        """Greedy decoding. Returns a list of token-id lists (one per image)."""
        self.eval()
        B = features.size(0)
        state = self.init_state(features)
        inp = torch.full((B, 1), start_idx, dtype=torch.long, device=features.device)
        outputs, finished = [], torch.zeros(B, dtype=torch.bool, device=features.device)
        for _ in range(max_len):
            out, state = self.lstm(self.embed(inp), state)
            nxt = self.fc(out.squeeze(1)).argmax(-1)
            outputs.append(nxt)
            finished |= nxt == end_idx
            if finished.all():
                break
            inp = nxt.unsqueeze(1)
        return torch.stack(outputs, 1).tolist()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
