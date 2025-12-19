import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import torch
import json
import re

from src.model import Encoder, Decoder, Attention, Seq2Seq

# ============================
# CONFIG
# ============================
MODEL_PATH = "models/seq2seq_demo.pt"
VOCAB_PATH = "data_clean/word2idx.json"

EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
DROPOUT = 0.2
MAX_LEN = 30

device = torch.device("cpu")

# ============================
# TOKENIZER
# ============================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s']", "", text)
    return text.split()

# ============================
# LOAD VOCAB
# ============================
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

idx2word = {idx: word for word, idx in word2idx.items()}

PAD = word2idx["<PAD>"]
SOS = word2idx["<SOS>"]
EOS = word2idx["<EOS>"]

VOCAB_SIZE = len(word2idx)

# ============================
# LOAD MODEL (ONCE)
# ============================
attention = Attention(HID_DIM)
encoder = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD)
decoder = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD, attention)

model = Seq2Seq(encoder, decoder, SOS, EOS, device).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================
# ENCODE / DECODE
# ============================
def encode(tokens):
    return [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]

def decode(token_ids):
    words = []
    last_word = None
    repeat_count = 0
    unk_count = 0

    for idx in token_ids:
        if idx == EOS:
            break
        if idx in (PAD, SOS):
            continue

        word = idx2word.get(idx, "<UNK>")

        # stop repetition
        if word == last_word:
            repeat_count += 1
            if repeat_count >= 2:
                break
        else:
            repeat_count = 0

        words.append(word)
        last_word = word

        if word == "<UNK>":
            unk_count += 1
            if unk_count >= 3:
                break

    return " ".join(words)

# ============================
# TRANSLATION FUNCTION
# ============================
def translate(text, src_lang, tgt_lang):
    src_token = f"<{src_lang}>"
    tgt_token = f"<{tgt_lang}>"

    tokens = [src_token, tgt_token] + tokenize(text)
    ids = encode(tokens)

    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.greedy_translate(src_tensor, max_len=MAX_LEN)

    return decode(output_ids)

# ============================
# GUI
# ============================
def on_translate():
    text = input_text.get("1.0", tk.END).strip()
    src = src_lang.get()
    tgt = tgt_lang.get()

    if not text:
        messagebox.showwarning("Input error", "Please enter text.")
        return

    if src == tgt:
        messagebox.showwarning("Input error", "Source and target languages must be different.")
        return

    result = translate(text, src, tgt)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, result)

# Window
root = tk.Tk()
root.title("Many-to-Many NLP Translator")
root.geometry("600x500")

# Title
title = tk.Label(root, text="Many-to-Many NLP Translator (From Scratch)", font=("Arial", 14, "bold"))
title.pack(pady=10)

# Input
tk.Label(root, text="Input Text:").pack(anchor="w", padx=10)
input_text = tk.Text(root, height=6)
input_text.pack(fill="x", padx=10, pady=5)

# Language selection
frame = tk.Frame(root)
frame.pack(pady=10)

tk.Label(frame, text="Source Language").grid(row=0, column=0, padx=10)
tk.Label(frame, text="Target Language").grid(row=0, column=1, padx=10)

src_lang = ttk.Combobox(frame, values=["EN", "FR", "ES"], state="readonly")
src_lang.current(0)
src_lang.grid(row=1, column=0, padx=10)

tgt_lang = ttk.Combobox(frame, values=["EN", "FR", "ES"], state="readonly")
tgt_lang.current(1)
tgt_lang.grid(row=1, column=1, padx=10)

# Button
translate_btn = tk.Button(root, text="Translate", command=on_translate)
translate_btn.pack(pady=10)

# Output
tk.Label(root, text="Translation:").pack(anchor="w", padx=10)
output_text = tk.Text(root, height=6, bg="#f5f5f5")
output_text.pack(fill="x", padx=10, pady=5)

# Run app
root.mainloop()
