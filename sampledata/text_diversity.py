import os
import numpy as np
import torch
import nltk
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2')

FILE_PAIRS = [
    ('sample_ZuluMax.txt', 'sample_ZuluMax.bin'),
    ('sample_YorubaMax.txt', 'sample_YorubaMax.bin'),
    ('sample_ZuluPrompt.txt', 'sample_ZuluPrompt.bin'),
    ('sample_YorubaPrompt.txt', 'sample_YorubaPrompt.bin'),
    ('sample_ZuluMini.txt', 'sample_ZuluMini.bin'),
    ('sample_YorubaMini.txt', 'sample_YorubaMini.bin')
]

def load_tokenized_data(bin_file):
    if not os.path.exists(bin_file):
        print(f"Tokenized file not found: {bin_file}, skipping...")
        return None
    return np.fromfile(bin_file, dtype=np.uint16)

def type_token_ratio(tokens):
    if tokens is None or len(tokens) == 0:
        return 0.0
    unique_types = set(tokens)
    return len(unique_types) / len(tokens)

def compute_semantic_similarity(stories):
    if len(stories) < 2:
        return 0.0  

    embeddings = model.encode(stories)

    similarity_matrix = cosine_similarity(embeddings)
    
    num_stories = len(stories)
    sim_values = [similarity_matrix[i, j] for i in range(num_stories) for j in range(i+1, num_stories)]
    
    return np.mean(sim_values) if sim_values else 0.0

def load_stories(txt_file):
    if not os.path.exists(txt_file):
        print(f"Story file not found: {txt_file}, skipping...")
        return []
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        return [story.strip() for story in f.read().split("---------------") if story.strip()]

def evaluate_diversity(txt_file, bin_file):
    stories = load_stories(txt_file)
    print(f"Loaded {len(stories)} stories from {txt_file}")
    tokens = load_tokenized_data(bin_file)

    ttr = type_token_ratio(tokens)
    semantic_sim = compute_semantic_similarity(stories)

    return {
        "type_token_ratio": ttr,
        "semantic_similarity": semantic_sim,
        "token_count": len(tokens) if tokens is not None else 0
    }

for txt_file, bin_file in FILE_PAIRS:
    scores = evaluate_diversity(txt_file, bin_file)
    
    print(f"Diversity scores for {txt_file}:")
    print(f"  Type-Token Ratio: {scores['type_token_ratio']:.4f}")
    print(f"  Semantic Similarity: {scores['semantic_similarity']:.4f}")
    print(f"  Token Count: {scores['token_count']}")
    print()
