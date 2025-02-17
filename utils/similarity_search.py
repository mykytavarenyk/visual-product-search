import os
import pickle
import numpy as np
from functools import lru_cache


def cosine_similarity(vec1, vec2, epsilon=1e-10):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + epsilon)


@lru_cache(maxsize=1)
def load_artifacts():
    """
    Load saved artifacts: embeddings, product info, KMeans model, and classifier.
    Cached for performance.
    """
    artifacts_dir = os.path.join("data", "artifacts")
    embeddings_path = os.path.join(artifacts_dir, "embeddings.pkl")
    product_info_path = os.path.join(artifacts_dir, "product_info.pkl")
    kmeans_path = os.path.join(artifacts_dir, "kmeans_model.pkl")
    classifier_path = os.path.join(artifacts_dir, "classifier.pkl")
    try:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        with open(product_info_path, "rb") as f:
            product_info = pickle.load(f)
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)
        with open(classifier_path, "rb") as f:
            clf = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError("Artifacts not found. Please run precompute_pipeline.py first.")
    return embeddings, product_info, kmeans, clf


def find_similar_products_in_cluster(query_embedding):
    """
    Given a query image's embedding, predict its cluster using the classifier and find
    all images within that cluster sorted by cosine similarity.

    Returns a list of dictionaries with the image path and similarity score.
    """
    embeddings, product_info, kmeans, clf = load_artifacts()

    # Convert query embedding to a numpy array
    query_vec = query_embedding.cpu().numpy().flatten() if hasattr(query_embedding,
                                                                   'cpu') else query_embedding.flatten()

    # Predict cluster label for the query image using the classifier
    cluster_label = clf.predict([query_vec])[0]

    # Identify indices of images that belong to the predicted cluster
    cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_label]
    if not cluster_indices:
        return []  # No images found in the predicted cluster

    # Compute cosine similarity for each image in the cluster
    results = []
    for i in cluster_indices:
        score = cosine_similarity(query_vec, embeddings[i])
        results.append({"image_path": product_info[i]["image_path"], "score": score})

    # Sort results by descending similarity score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
