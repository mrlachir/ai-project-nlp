from datasets import load_dataset

# 1. English - French
dataset_en_fr = load_dataset("opus_books", "en-fr")

# 2. English - Spanish
dataset_en_es = load_dataset("opus_books", "en-es")

# 3. French - Spanish (Direct!)
dataset_fr_es = load_dataset("opus_books", "es-fr")

# Example: Print a sample
print(dataset_fr_es['train'][10]) 
# Output: {'id': '11', 'translation': {'es': 'El fue a mi casa.', 'fr': 'Il est allé chez moi.'}}