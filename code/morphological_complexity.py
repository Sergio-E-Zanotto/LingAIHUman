#!/usr/bin/env python3.10
# coding: utf-8

import spacy
import pandas as pd
import numpy as np
import random
from itertools import combinations

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

def analyze_text_with_spacy(text):
    """
    Analyze the text using spaCy to extract words, lemmas, and POS tags.
    Returns a DataFrame with tokens, lemmas, and POS.
    """
    doc = nlp(text)
    data = {
        "Word": [token.text for token in doc],
        "Lemma": [token.lemma_ for token in doc],
        "POS": [token.pos_ for token in doc],
    }
    return pd.DataFrame(data)

def identify_exponences(df, word_class):
    """
    Identify the default stem (lemma) and group word forms by lemma.
    Returns a DataFrame with lemmas and their corresponding word forms.
    """
    filtered_df = df[df["POS"] == word_class]
    grouped = filtered_df.groupby("Lemma")["Word"].apply(set).reset_index()
    return grouped

def calculate_within_subset_variety(subset):
    """
    Calculate the within-subset variety by counting unique exponences.
    """
    subset_exponences = set().union(*subset)
    return len(subset_exponences)

def calculate_between_subset_diversity(subset1, subset2):
    """
    Calculate the between-subset diversity by finding the symmetric difference
    of exponences between two subsets.
    """
    exponences1 = set().union(*subset1)
    exponences2 = set().union(*subset2)
    return len(exponences1.symmetric_difference(exponences2))

def calculate_mci(data, word_class, subset_size, trials=100):
    """
    Calculate the Morphological Complexity Index (MCI).
    """
    # Identify exponences for the given word class
    exponences = identify_exponences(data, word_class)
    
    # Ensure subset size does not exceed the available data
    max_subset_size = min(subset_size, len(exponences))
    if max_subset_size == 0:
        raise ValueError(f"No data available for word class: {word_class}")

    within_varieties = []
    between_diversities = []
    
    for _ in range(trials):
        # Randomly sample two subsets
        subset1 = exponences.sample(max_subset_size)["Word"].tolist()
        subset2 = exponences.sample(max_subset_size)["Word"].tolist()
        
        # Calculate within-subset variety
        within_varieties.append(calculate_within_subset_variety(subset1))
        
        # Calculate between-subset diversity
        between_diversities.append(calculate_between_subset_diversity(subset1, subset2))
    
    # Compute averages
    within_avg = np.mean(within_varieties)
    between_avg = np.mean(between_diversities)
    
    # Calculate MCI
    mci = (within_avg + between_avg / 2) - 1
    return mci

def analyze_dataframe(df, word_classes, subset_size=5, trials=100):
    """
    Analyze each row of the DataFrame and calculate MCI for specified word classes.
    Returns a new DataFrame with MCIs for each row and word class.
    """
    results = []
    for index, row in df.iterrows():
        text = row["Text"]
        # Analyze the text with spaCy
        processed_data = analyze_text_with_spacy(text)
        row_results = {"Row": index}
        for word_class in word_classes:
            try:
                mci = calculate_mci(processed_data, word_class, subset_size, trials)
                row_results[f"MCI_{word_class}"] = mci
            except ValueError:
                row_results[f"MCI_{word_class}"] = None  # Handle cases with insufficient data
        results.append(row_results)
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example: Replace this with your actual DataFrame loading logic
    
    df = pd.read_csv("raid.csv") #choose either official_text_ai.csv or raid.csv
    
    # Specify the word classes to analyze (e.g., VERB, NOUN)
    word_classes = ["VERB", "NOUN"]
    
    # Analyze the DataFrame
    results_df = analyze_dataframe(df, word_classes, subset_size=5, trials=100)
    
    # Save the results to a CSV file
    results_df.to_csv("raid_mci_results.csv", index=False)
    print("Analysis complete. Results saved to 'mci_results.csv'.")
