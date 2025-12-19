import streamlit as st
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
@st.cache_resource
def load_model():
    attention = Attention(HID_DIM)
    encoder = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD)
    decoder = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, PAD, attention)

    model = Seq2Seq(encoder, decoder, SOS, EOS, device).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# ============================
# ENCODE / DECODE
# ============================
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
            if unk_count >= 5:
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
# STREAMLIT UI
# ============================
st.title("🌍 Many-to-Many NLP Translator (From Scratch)")

st.markdown("""
This system demonstrates a **many-to-many neural machine translation model**
trained **from scratch** using a Seq2Seq LSTM with attention.
""")

text = st.text_area("Enter text to translate", height=120)

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source language", ["EN", "FR", "ES"])
with col2:
    tgt_lang = st.selectbox("Target language", ["EN", "FR", "ES"])

if st.button("Translate"):
    if src_lang == tgt_lang:
        st.warning("Source and target languages must be different.")
    elif not text.strip():
        st.warning("Please enter some text.")
    else:
        result = translate(text, src_lang, tgt_lang)
        st.subheader("Translation")
        st.success(result)
