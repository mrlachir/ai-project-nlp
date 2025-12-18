import os
import json
import torch
import torch.nn as nn

from src.dataset import make_dataloader
from src.model import Encoder, Decoder, Attention, Seq2Seq

# ============================
# CONFIG (CPU SAFE)
# ============================
DATA_PATH = "data_clean/many_to_many_dataset.csv"
VOCAB_PATH = "data_clean/word2idx.json"
MODEL_DIR = "models"

EPOCHS = 1                  # ONE epoch is enough for demo
BATCH_SIZE = 4              # VERY important for CPU
MAX_LEN = 20                # shorter sequences = faster
EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
DROPOUT = 0.2
LR = 0.001
TEACHER_FORCING = 0.5

MAX_BATCHES_PER_EPOCH = 300  # <<< KEY SPEED FIX

# ============================
# SETUP
# ============================
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cpu")

# ============================
# LOAD VOCAB
# ============================
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

VOCAB_SIZE = len(word2idx)
PAD = word2idx["<PAD>"]
SOS = word2idx["<SOS>"]
EOS = word2idx["<EOS>"]

print("Vocabulary size:", VOCAB_SIZE)

# ============================
# DATA LOADER
# ============================
train_loader, _ = make_dataloader(
    DATA_PATH,
    VOCAB_PATH,
    batch_size=BATCH_SIZE,
    max_len=MAX_LEN,
    shuffle=True
)

print("Batches per epoch (full):", len(train_loader))
print("Batches per epoch (used):", MAX_BATCHES_PER_EPOCH)

# ============================
# MODEL
# ============================
attention = Attention(HID_DIM)
encoder = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD)
decoder = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD, attention)

model = Seq2Seq(encoder, decoder, SOS, EOS, device).to(device)

# ============================
# OPTIMIZER & LOSS
# ============================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

# ============================
# TRAINING FUNCTION
# ============================
def train_epoch(model, loader):
    model.train()
    epoch_loss = 0

    for i, (src, tgt) in enumerate(loader):
        if i >= MAX_BATCHES_PER_EPOCH:
            break

        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        output = model(src, tgt, TEACHER_FORCING)

        # output: [B, T, V]
        # tgt:    [B, T]
        output_dim = output.shape[-1]

        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}/{MAX_BATCHES_PER_EPOCH} | Loss: {loss.item():.4f}")

    return epoch_loss / (i + 1)

# ============================
# TRAIN
# ============================
print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader)

    print(f"\nEpoch {epoch}/{EPOCHS} | Avg Loss: {loss:.4f}")

    torch.save(
        model.state_dict(),
        f"{MODEL_DIR}/seq2seq_demo.pt"
    )

print("\nTraining finished.")
print("Model saved to models/seq2seq_demo.pt")
