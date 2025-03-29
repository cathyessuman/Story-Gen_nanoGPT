import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance

# Ensure necessary NLTK tokenizer is available
nltk.download('punkt')

# Load the Sentence Transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_stories_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Text file not found: {file_path}, skipping...")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        stories = f.read().strip().split("\n\n")  
        stories = [s.strip() for s in stories if len(s.strip()) > 10]  

    if len(stories) == 0:
        print(f"Warning: {file_path} contains no valid stories!")

    return stories

def load_tokenized_data(bin_file):
    if not os.path.exists(bin_file):
        print(f"Binary file not found: {bin_file}, skipping...")
        return np.array([])
    return np.fromfile(bin_file, dtype=np.uint16)

def type_token_ratio(tokens):
    types = set(tokens)
    return len(types) / len(tokens)


def semantic_similarity_cosine(stories):
    if len(stories) < 2:
        return 0.0 

    embeddings = st_model.encode(stories)

    similarity_scores = []
    num_stories = len(embeddings)

    for i in range(num_stories):
        for j in range(i + 1, num_stories):
            sim_score = 1 - distance.cosine(embeddings[i], embeddings[j])
            similarity_scores.append(sim_score)

    return np.mean(similarity_scores) if similarity_scores else 0.0

def evaluate_file(txt_file, bin_file):
    stories = load_stories_from_file(txt_file)
    tokens = load_tokenized_data(bin_file)

    if len(stories) == 0 or len(tokens) == 0:
        return None

    ttr = type_token_ratio(tokens)
    cosine_sim = semantic_similarity_cosine(stories)
    token_count = len(tokens)

    print(f"\nDiversity scores for {txt_file}:")
    print(f"Type-Token Ratio: {ttr:.4f}")
    print(f"Semantic Similarity: {cosine_sim:.4f}")
    print(f"Token Count: {token_count}\n")


file_pairs = [
    ('eval_ZuluMax.txt', 'eval_ZuluMax.bin'),
    ('eval_YorubaMax.txt', 'eval_YorubaMax.bin'),
    ('eval_ZuluPrompted.txt', 'eval_ZuluPrompted.bin'),
    ('eval_YorubaPrompted.txt', 'eval_YorubaPrompted.bin'),
    ('eval_ZuluMini.txt', 'eval_ZuluMini.bin'),
    ('eval_YorubaMini.txt', 'eval_YorubaMini.bin')
]

for txt_file, bin_file in file_pairs:
    evaluate_file(txt_file, bin_file)
