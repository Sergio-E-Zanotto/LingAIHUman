#!/usr/bin/env python3.10
# coding: utf-8

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
# Load the pre-trained models
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the dataset
df = pd.read_csv("raid.csv", encoding="UTF-8") #choose either official_text_ai.csv or raid.csv
print("DataFrame loaded successfully.")

# Function to compute semantic distance using sentences
def compute_semantic_distance_sentences(text):
    sentences = sent_tokenize(text)
    embeddings = sentence_model.encode(sentences)
    distances = []
    for i in range(len(embeddings) - 1):
        for j in range(i + 1, len(embeddings)):
            distance = util.pytorch_cos_sim(embeddings[i], embeddings[j])
            distances.append(distance.item())
    if distances:
        average_distance = sum(distances) / len(distances)
    else:
        average_distance = 0
    return average_distance

# Calculate semantic distance for each text using sentences and store in DataFrame
df['Semantic_similarity'] = df['Text'].apply(compute_semantic_distance_sentences)

print("Semantic analysis completed.")

# Save the updated DataFrame to a new CSV file
df.to_csv("raid_semantic_results.csv", index=False)
print("CSV file created successfully with updated semantic distances.")
