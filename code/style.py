#!/usr/bin/env python3.10
# coding: utf-8

import pandas as pd
import logging
import torch
from sentence_transformers import SentenceTransformer

# Load StyleDistance model
style_model = SentenceTransformer('StyleDistance/styledistance')

# Configure logging
logging.basicConfig(level=logging.ERROR)
print("Libraries and StyleDistance model imported successfully.")

def compute_style_embedding(text):
    """
    Computes the style embedding for a given text using the StyleDistance model.
    """
    try:
        embedding = style_model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy().tolist() if torch.is_tensor(embedding) else embedding.tolist()
    except Exception as e:
        logging.error(f"Exception {e} occurred for text: {text}")
        return None

# Read the CSV file
df = pd.read_csv("raid.csv", encoding="UTF-8")
print("DataFrame loaded successfully.")

# Compute Style Embeddings
df["Style_Embedding"] = df["Text"].apply(compute_style_embedding)

print("Style embedding computation completed.")

# Save the results
df.to_csv("raid_style.csv", index=False)

print("Analysis complete. Results saved to 'raid_result/raid_results_with_style.csv'.")
