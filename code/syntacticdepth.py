#!/usr/bin/env python3.10
# coding: utf-8

import os
import sys
import warnings
import logging
import pandas as pd
import spacy

# Suppress specific warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'")

# Configure logging to output both to the console and a file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for logging
file_handler = logging.FileHandler('syntacticdepth.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")
print("Libraries and NLP model imported successfully.")

def calculate_depth(token):
    """
    Recursively calculates the depth of the dependency tree starting at the given token.
    A leaf node has a depth of 1; otherwise, the depth is 1 plus the maximum depth of its children.
    """
    if not list(token.children):
        return 1
    else:
        return 1 + max(calculate_depth(child) for child in token.children)

def analyze_dependency_depth(text):
    """
    Analyzes the dependency tree depth (i.e., hierarchical syntactic complexity) of the provided text.
    It calculates the depth of each sentence's dependency tree (from the root to the deepest leaf)
    and returns the average depth across all sentences.
    """
    try:
        doc = nlp(text)
        total_depth = 0
        sentence_count = 0

        for sentence in doc.sents:
            # Identify the root of the sentence (token whose head is itself)
            root_candidates = [token for token in sentence if token.head == token]
            if not root_candidates:
                logging.warning(f"No root found in sentence: {sentence.text}")
                continue
            root = root_candidates[0]
            depth = calculate_depth(root)
            total_depth += depth
            sentence_count += 1

        average_depth = total_depth / sentence_count if sentence_count > 0 else 0
        return average_depth
    except AssertionError:
        logging.error(f"AssertionError: Mismatch in sentence and tree counts for text: {text}")
        return "nan"
    except Exception as e:
        logging.error(f"Exception {e} occurred for text: {text}")
        return "nan"

# Read the CSV file
df = pd.read_csv("raid.csv", encoding="UTF-8")  # Choose either official_text_ai.csv or raid.csv
print("DataFrame loaded successfully.")

# Calculate dependency tree depth (syntactic complexity) for each text and store in the DataFrame
df['Syntactic_Depth'] = df['Text'].apply(analyze_dependency_depth)
print("Syntactic analysis completed.")

output_filename = "raid_syntactic_results.csv"
df.to_csv(output_filename, index=False)
print(f"Analysis complete. Results saved to '{output_filename}'.")
