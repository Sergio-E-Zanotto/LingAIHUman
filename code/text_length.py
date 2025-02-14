#!/usr/bin/env python3.10
# coding: utf-8

import os
import pandas as pd
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Ensure result directory exists
result_dir = "raid_result"
os.makedirs(result_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv("raid.csv", encoding="UTF-8")

### **Step 1: Combine Model Details into a Single Identifier**
def combine_model(row):
    parts = [str(row["model"])]
    
    if pd.notnull(row["decoding"]) and str(row["decoding"]).strip():
        parts.append(str(row["decoding"]).strip())
        
    if pd.notnull(row["repetition_penalty"]) and str(row["repetition_penalty"]).strip():
        parts.append(str(row["repetition_penalty"]).strip())
        
    return "_".join(parts)

df["model"] = df.apply(combine_model, axis=1)

### **Step 2: Compute Text Length**
def compute_text_length(text):
    doc = nlp(str(text))
    return len([token for token in doc if token.is_alpha])

df["Text_Length"] = df["Text"].apply(compute_text_length)

### **Step 3: Compute the Average Text Length per Combined Model**
text_length_per_model = df.groupby("model")["Text_Length"].mean().reset_index()
text_length_per_model.columns = ["model", "Average_Text_Length"]

# Save to CSV
text_length_per_model.to_csv(f"{result_dir}/average_text_length_per_model.csv", index=False)
print(f"Saved: {result_dir}/average_text_length_per_model.csv")

### **Step 4: Compute Average Text Length per Model per Domain**
if "domain" in df.columns:
    text_length_per_domain = df.groupby(["domain", "model"])["Text_Length"].mean().reset_index()
    text_length_per_domain.columns = ["domain", "model", "Average_Text_Length"]

    # Save to CSV
    text_length_per_domain.to_csv(f"{result_dir}/average_text_length_per_model_per_domain.csv", index=False)
    print(f"Saved: {result_dir}/average_text_length_per_model_per_domain.csv")

print("Average text length calculation completed.")
