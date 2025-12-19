import json
import torch
import re

from src.model import Encoder, Decoder, Attention, Seq2Seq

# ============================
# CONFIG
# ============================
MODEL_PATH = "models/seq2seq_demo.pt"
VOCAB_PATH = "data_clean/word2idx.json"
MAX_LEN = 30

EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
DROPOUT = 0.2

device = torch.device("cpu")

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
# TOKENIZER (same as training)
# ============================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s']", "", text)
    return text.split()

def encode(tokens):
    return [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]

def decode(token_ids):
    words = []
    unk_count = 0

    for idx in token_ids:
        if idx == EOS:
            break
        if idx in (PAD, SOS):
            continue

        word = idx2word.get(idx, "<UNK>")
        words.append(word)

        if word == "<UNK>":
            unk_count += 1
            if unk_count >= 5:  # stop early
                break

    return " ".join(words)

# ============================
# LOAD MODEL
# ============================
attention = Attention(HID_DIM)
encoder = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD)
decoder = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD, attention)

model = Seq2Seq(encoder, decoder, SOS, EOS, device).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================
# TRANSLATION FUNCTION
# ============================
def translate(text, src_lang, tgt_lang):
    """
    src_lang, tgt_lang: 'en', 'fr', or 'es'
    """
    src_token = f"<{src_lang.upper()}>"
    tgt_token = f"<{tgt_lang.upper()}>"

    tokens = [src_token, tgt_token] + tokenize(text)
    ids = encode(tokens)
    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    output_ids = model.greedy_translate(src_tensor, max_len=MAX_LEN)
    return decode(output_ids)

# ============================
# DEMO
# ============================
if __name__ == "__main__":
    print("\n=== MANY-TO-MANY TRANSLATION DEMO ===\n")

    while True:
        text = input("Enter text (or 'quit'): ")
        if text.lower() == "quit":
            break

        src = input("Source language (en/fr/es): ").lower()
        tgt = input("Target language (en/fr/es): ").lower()

        result = translate(text, src, tgt)
        print("\nTranslation:", result)
        print("-" * 40)
