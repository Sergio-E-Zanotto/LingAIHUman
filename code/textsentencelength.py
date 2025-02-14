#!/usr/bin/env python3.10
# coding: utf-8

import os
import sys
import warnings
import logging
import pandas as pd
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

def compute_text_stats_spacy(text):
    # Ensure the text is a string
    if not isinstance(text, str):
        return pd.Series({"Text_Length": np.nan, "Avg_Sentence_Length": np.nan})
    
    doc = nlp(text)
    
    # Count the number of tokens that are words (exclude punctuation and spaces)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    text_length = len(words)
    
    # Use spaCy's sentence segmentation
    sentences = list(doc.sents)
    
    if sentences:
        # For each sentence, count tokens that are words (exclude punctuation and spaces)
        sentence_lengths = [
            len([token for token in sent if not token.is_punct and not token.is_space])
            for sent in sentences
        ]
        avg_sentence_length = np.mean(sentence_lengths)
    else:
        avg_sentence_length = 0
        
    return pd.Series({"Text_Length": text_length, "Avg_Sentence_Length": avg_sentence_length})

# Read the CSV file
df = pd.read_csv("raid.csv", encoding="UTF-8")  # Choose either official_text_ai.csv or raid.csv
print("DataFrame loaded successfully.")

# Assuming your DataFrame is called 'df' and has a column 'Text'
df[['Text_Length', 'Avg_Sentence_Length']] = df['Text'].apply(compute_text_stats_spacy)


output_filename = "raid_textsentence_results.csv"
df.to_csv(output_filename, index=False)
print(f"Analysis complete. Results saved to '{output_filename}'.")
