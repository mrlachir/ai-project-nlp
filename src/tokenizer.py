import pandas as pd
from collections import Counter
import re

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = "data_clean/many_to_many_dataset.csv"
VOCAB_SIZE = 30000   # was 30000
SPECIAL_TOKENS = [
    "<PAD>", "<SOS>", "<EOS>", "<UNK>",
    "<EN>", "<FR>", "<ES>"
]

# ----------------------------
# Basic tokenizer
# ----------------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s']", "", text)
    return text.split()

# ----------------------------
# Build vocabulary
# ----------------------------
def build_vocab(df, vocab_size):
    counter = Counter()

    for text in df["src_text"]:
        counter.update(tokenize(text))

    for text in df["tgt_text"]:
        counter.update(tokenize(text))

    most_common = counter.most_common(vocab_size - len(SPECIAL_TOKENS))

    vocab = SPECIAL_TOKENS + [word for word, _ in most_common]

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word

# ----------------------------
# Encode sentence
# ----------------------------
def encode_sentence(tokens, word2idx):
    return [
        word2idx.get(token, word2idx["<UNK>"])
        for token in tokens
    ]

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("Building vocabulary...")
    word2idx, idx2word = build_vocab(df, VOCAB_SIZE)

    print("Vocabulary size:", len(word2idx))

    # Save vocab
    import json
    with open("data_clean/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    with open("data_clean/idx2word.json", "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False, indent=2)

    print("Vocabulary saved.")
