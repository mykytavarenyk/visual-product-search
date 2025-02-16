import os
import pickle
from PIL import Image, UnidentifiedImageError
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Update dataset_dir to point to your image folder
dataset_dir = os.path.join("data", "dataset", "test_task_data")
embeddings_dir = os.path.join("data", "embeddings")

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
        embeddings.append(embedding.cpu().numpy().flatten())
        product_info.append({"image_path": image_path, "filename": filename})
        
os.makedirs(embeddings_dir, exist_ok=True)
with open(os.path.join(embeddings_dir, "dataset_embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)

with open(os.path.join(embeddings_dir, "product_info.pkl"), "wb") as f:
    pickle.dump(product_info, f)

print("Embeddings have been computed and saved successfully!")
