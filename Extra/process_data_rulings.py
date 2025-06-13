import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Read the CSV file
print("Loading dataset...")
df = pd.read_csv('swiss_rulings_with_translations.csv')

# Calculate tokens for each text
print("Calculating tokens...")
token_counts = []
for text in tqdm(df['full_text_en']):
    if pd.isna(text):
        token_counts.append(0)
    else:
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))

# Calculate statistics
avg_tokens = np.mean(token_counts)
median_tokens = np.median(token_counts)
min_tokens = np.min(token_counts)
max_tokens = np.max(token_counts)

print("\nToken Statistics for full_text_en:")
print(f"Average tokens: {avg_tokens:.2f}")
print(f"Median tokens: {median_tokens:.2f}")
print(f"Minimum tokens: {min_tokens}")
print(f"Maximum tokens: {max_tokens}")
print(f"Total texts analyzed: {len(token_counts)}")
print(f"Texts with 0 tokens (empty/NaN): {sum(1 for x in token_counts if x == 0)}")

