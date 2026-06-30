"""
Created by Marián Tarageľ (xtarag01)
"""

import pandas as pd
from collections import Counter
import re

df = pd.read_csv('~/datasets/Contrastive_Dent_Dataset/splits/processed-split.csv')

print("--- Basic Statistics ---")
print(f"Total Utterances: {len(df)}")

df['context_id'] = df['source_uid'] + " -> " + df['target_uid']
n_contexts = df['context_id'].nunique()
print(f"Distinct Contexts: {n_contexts}")
print(f"Average Utterances per Context: {len(df) / n_contexts:.2f}")

print(f"Unique Utterances: {df['utterance'].nunique()}")
print(f"Unique Source Models: {df['source_uid'].nunique()}")
print(f"Unique Target Models: {df['target_uid'].nunique()}")
a = pd.concat([df['source_uid'], df['target_uid']], ignore_index=True)
print(f"Total Unique Shapes: {a.nunique()}")

def get_op(row):
    if row['target_object_class'] == 'aligned':
        return 'align'
    if 'extracted' in row['target_uid']:
        return 'extract'
    return 'replace'

df['operation'] = df.apply(get_op, axis=1)
print("\n--- Operation Distribution (%) ---")
print(df['operation'].value_counts(normalize=True) * 100)

def extract_tooth(uid):
    match = re.search(r'_(\d{2})_', uid)
    return match.group(1) if match else "None/Align"

df['tooth_id'] = df.apply(lambda x: extract_tooth(x['target_uid']) if x['operation'] == 'extract' 
                          else (extract_tooth(x['source_uid']) if x['operation'] == 'replace' else "Align"), axis=1)

print("\n--- Teeth Involved ---")
tooth_counts = df[df['tooth_id'] != "Align"]['tooth_id'].value_counts()
print(tooth_counts.head(32))

all_tokens = []
df['token_list'] = df['tokens'].apply(eval)
for tokens in df['token_list']:
    all_tokens.extend([t.lower() for t in tokens])

vocab = set(all_tokens)
print("\n--- Language Stats ---")
print(f"Vocabulary Size: {len(vocab)}")
print(f"Avg Utterance Length: {df['token_list'].apply(len).mean():.2f} words")
print(f"Top 10 words: {Counter(all_tokens).most_common(10)}")