import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import re


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s']", "", text)
    return text.split()


class TranslationDataset(Dataset):
    def __init__(self, csv_path, word2idx_path, max_len=30):
        self.df = pd.read_csv(csv_path)
        with open(word2idx_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        self.max_len = max_len

        # special indices
        self.pad = self.word2idx["<PAD>"]
        self.sos = self.word2idx["<SOS>"]
        self.eos = self.word2idx["<EOS>"]
        self.unk = self.word2idx["<UNK>"]

    def encode(self, tokens):
        return [self.word2idx.get(tok, self.unk) for tok in tokens]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        src_lang = f"<{row['src_lang'].upper()}>"
        tgt_lang = f"<{row['tgt_lang'].upper()}>"

        src_tokens = [src_lang, tgt_lang] + tokenize(row["src_text"])
        tgt_tokens = ["<SOS>"] + tokenize(row["tgt_text"]) + ["<EOS>"]

        src_ids = self.encode(src_tokens)[: self.max_len]
        tgt_ids = self.encode(tgt_tokens)[: self.max_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, : len(src)] = src
        padded_tgt[i, : len(tgt)] = tgt

    return padded_src, padded_tgt


def make_dataloader(csv_path, word2idx_path, batch_size=32, max_len=30, shuffle=True):
    ds = TranslationDataset(csv_path, word2idx_path, max_len=max_len)
    pad_idx = ds.pad

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_idx),
    )
    return loader, ds
