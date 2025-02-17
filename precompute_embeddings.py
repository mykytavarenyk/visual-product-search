import os
import pickle
from PIL import Image, UnidentifiedImageError
import torch
import clip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

dataset_dir = os.path.join("data", "dataset", "test_task_data")
artifacts_dir = os.path.join("data", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

embeddings = []
product_info = []

for filename in os.listdir(dataset_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        image_path = os.path.join(dataset_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Skipping invalid image file: {filename}")
            continue
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
        embedding_np = embedding.cpu().numpy().flatten()
        embeddings.append(embedding_np)
        product_info.append({"image_path": image_path, "filename": filename})

embeddings = np.array(embeddings)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(embeddings, cluster_labels)

with open(os.path.join(artifacts_dir, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)

with open(os.path.join(artifacts_dir, "product_info.pkl"), "wb") as f:
    pickle.dump(product_info, f)

with open(os.path.join(artifacts_dir, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)

with open(os.path.join(artifacts_dir, "classifier.pkl"), "wb") as f:
    pickle.dump(clf, f)

print("Artifacts saved successfully!")
