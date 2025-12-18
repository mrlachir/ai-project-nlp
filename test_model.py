import torch
import json
from src.dataset import make_dataloader
from src.model import Encoder, Decoder, Attention, Seq2Seq

device = torch.device("cpu")

with open("data_clean/word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)

VOCAB_SIZE = len(word2idx)
PAD = word2idx["<PAD>"]
SOS = word2idx["<SOS>"]
EOS = word2idx["<EOS>"]

loader, ds = make_dataloader("data_clean/many_to_many_dataset.csv", "data_clean/word2idx.json", batch_size=8)

attn = Attention(hid_dim=256)
enc = Encoder(VOCAB_SIZE, emb_dim=128, hid_dim=256, n_layers=1, dropout=0.2, pad_idx=PAD)
dec = Decoder(VOCAB_SIZE, emb_dim=128, hid_dim=256, n_layers=1, dropout=0.2, pad_idx=PAD, attention=attn)

model = Seq2Seq(enc, dec, SOS, EOS, device).to(device)

src, tgt = next(iter(loader))
out = model(src, tgt, teacher_forcing_ratio=0.5)

print("src shape:", src.shape)
print("tgt shape:", tgt.shape)
print("out shape:", out.shape)
