#!/usr/bin/env python3.10
# coding: utf-8

import os
import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Ensure result directory exists
result_dir = "raid_result"
os.makedirs(result_dir, exist_ok=True)

# Read data
df = pd.read_csv("raid.csv", encoding="utf-8")

# Function to combine model details
def combine_model(row):
    parts = [str(row["model"])]

    if pd.notnull(row["decoding"]) and str(row["decoding"]).strip():
        parts.append(str(row["decoding"]).strip())
        
    if pd.notnull(row["repetition_penalty"]) and str(row["repetition_penalty"]).strip():
        parts.append(str(row["repetition_penalty"]).strip())
        
    return "_".join(parts)

df["model"] = df.apply(combine_model, axis=1)

# Function to compute sentence lengths
def count_sentence_lengths(text):
    """
    Extracts sentence lengths (in words) from text.
    """
    doc = nlp(str(text))  
    return [len(sent) for sent in doc.sents]  

df["Sentence_Lengths"] = df["Text"].apply(count_sentence_lengths)

# Flatten sentence lengths per model
model_sentence_lengths = {}
for model, group in df.groupby("model"):
    all_lengths = [length for sublist in group["Sentence_Lengths"].dropna() for length in sublist]
    model_sentence_lengths[model] = all_lengths

# Prepare DataFrame for model-level sentence length distribution
length_distribution_model = pd.DataFrame()

for model, lengths in model_sentence_lengths.items():
    if len(lengths) == 0:
        continue  

    length_counts = pd.Series(lengths).value_counts().sort_index()
    normalized_freq = length_counts / length_counts.sum()
    
    model_df = pd.DataFrame({
        "Sentence_Length": normalized_freq.index,
        "Normalized_Frequency": normalized_freq.values,
        "Model": model
    })
    
    length_distribution_model = pd.concat([length_distribution_model, model_df], ignore_index=True)

# Save per-model distribution to CSV
length_distribution_model.to_csv(f"{result_dir}/sentence_length_distribution_per_model.csv", index=False)
print(f"Saved: {result_dir}/sentence_length_distribution_per_model.csv")

### **Step 2: Compute Sentence Length Distributions Per Domain**
if "domain" in df.columns:
    length_distribution_domain = pd.DataFrame()

    for domain, domain_df in df.groupby("domain"):
        domain_sentence_lengths = {}

        for model, group in domain_df.groupby("model"):
            all_lengths = [length for sublist in group["Sentence_Lengths"].dropna() for length in sublist]
            domain_sentence_lengths[(domain, model)] = all_lengths

        for (domain, model), lengths in domain_sentence_lengths.items():
            if len(lengths) == 0:
                continue  

            length_counts = pd.Series(lengths).value_counts().sort_index()
            normalized_freq = length_counts / length_counts.sum()

            domain_model_df = pd.DataFrame({
                "Domain": domain,
                "Sentence_Length": normalized_freq.index,
                "Normalized_Frequency": normalized_freq.values,
                "Model": model
            })

            length_distribution_domain = pd.concat([length_distribution_domain, domain_model_df], ignore_index=True)

    # Save per-domain distribution to CSV
    length_distribution_domain.to_csv(f"{result_dir}/sentence_length_distribution_per_model_per_domain.csv", index=False)
    print(f"Saved: {result_dir}/sentence_length_distribution_per_model_per_domain.csv")

print("Sentence length distribution calculations completed.")
