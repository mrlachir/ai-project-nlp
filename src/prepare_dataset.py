from datasets import load_dataset
import pandas as pd

MAX_SAMPLES_PER_PAIR = 30000
OUTPUT_PATH = "data_clean/many_to_many_dataset.csv"


def extract_pairs(dataset, lang1, lang2, limit):
    rows = []

    subset = dataset["train"].select(range(min(limit, len(dataset["train"]))))

    for example in subset:
        trans = example["translation"]

        text1 = trans[lang1].strip()
        text2 = trans[lang2].strip()

        if text1 and text2:
            # lang1 -> lang2
            rows.append({
                "src_lang": lang1,
                "tgt_lang": lang2,
                "src_text": text1,
                "tgt_text": text2
            })

            # lang2 -> lang1
            rows.append({
                "src_lang": lang2,
                "tgt_lang": lang1,
                "src_text": text2,
                "tgt_text": text1
            })

    return rows


print("Loading datasets...")
dataset_en_fr = load_dataset("opus_books", "en-fr")
dataset_en_es = load_dataset("opus_books", "en-es")
dataset_es_fr = load_dataset("opus_books", "es-fr")

data = []

print("Processing EN <-> FR")
data.extend(extract_pairs(dataset_en_fr, "en", "fr", MAX_SAMPLES_PER_PAIR))

print("Processing EN <-> ES")
data.extend(extract_pairs(dataset_en_es, "en", "es", MAX_SAMPLES_PER_PAIR))

print("Processing ES <-> FR")
data.extend(extract_pairs(dataset_es_fr, "es", "fr", MAX_SAMPLES_PER_PAIR))

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("DONE")
print("Total samples:", len(df))
print(df.head())
