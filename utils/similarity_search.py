import os
import pickle
import numpy as np
from functools import lru_cache

def cosine_similarity(vec1, vec2):
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + epsilon)

@lru_cache(maxsize=1)
def load_embeddings_and_info():
    """
    Load dataset embeddings and product info from disk.
    Cached for performance optimization.
    """
    embeddings_path = os.path.join("data", "embeddings", "dataset_embeddings.pkl")
    product_info_path = os.path.join("data", "embeddings", "product_info.pkl")
    try:
        with open(embeddings_path, "rb") as f:
            dataset_embeddings = pickle.load(f)
        with open(product_info_path, "rb") as f:
            product_info = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Embeddings files not found. Please run precompute_embeddings.py first.")
    return dataset_embeddings, product_info

def find_similar_products(query_embedding, top_n=5, filter_extension="All"):
    """
    Find similar products based on the query embedding.
    :param query_embedding: Embedding of the query image.
    :param top_n: Number of top similar products to return.
    :param filter_extension: Filter results by image file extension.
    """
    dataset_embeddings, product_info = load_embeddings_and_info()
    # Convert query_embedding to numpy array if needed
    query_vec = query_embedding.cpu().numpy().flatten() if hasattr(query_embedding, 'cpu') else query_embedding.flatten()
    scores = []
    for info, emb in zip(product_info, dataset_embeddings):
        # Apply file extension filter if set
        if filter_extension != "All" and not info['image_path'].lower().endswith(filter_extension):
            continue
        score = cosine_similarity(query_vec, emb)
        scores.append((info, score))
    # Sort by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    results = [{"image_path": item[0]['image_path'], "score": item[1]} for item in scores[:top_n]]
    return results
