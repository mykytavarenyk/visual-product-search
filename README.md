# Visual Product Search - Project Report

## 1. Technical Approach

**Feature Extraction:**
- Utilizes the pre-trained CLIP model (ViT-B/32) from OpenAI to extract high-dimensional image embeddings.

**Embedding & Clustering:**
- Embeddings computed for all images
- KMeans clustering is applied to the embeddings, assigning a pseudo-label (cluster ID) to each image.
- This grouping enables the system to narrow down search results to a specific cluster.

**Classifier Training:**
- A logistic regression classifier is trained using the computed embeddings along with their assigned cluster labels.
- The classifier is used to predict the cluster for a new query image, ensuring that the search is performed within the most relevant group.

**Similarity Search Strategy:**
- For a given query image (optionally cropped and preprocessed), the system computes its CLIP embedding.
- The trained classifier predicts the cluster (pseudo-label) of the query.
- All images within the predicted cluster are compared with the query image using cosine similarity.
- The results are returned sorted in descending order by similarity score.

---

## 2. Project Structure and Architecture

```
visual-product-search/
│
├── data/
│   ├── dataset/
│   │   └── test_task_data/        # Folder containing product images dataset
│   └── artifacts/                 # Saved artifacts (embeddings, product info, KMeans & classifier models)
│
├── models/
│   └── bclip_model.py             # Loads the CLIP model and defines the get_embedding() function
│
├── static/                        # Contains static files (e.g., CSS for custom styling)
│
├── utils/
│   ├── image_utils.py             # Contains the enhanced preprocess_image() function
│   └── similarity_search.py       # Implements similarity search; loads artifacts and returns similar products
│
├── app.py                         # The main Streamlit app (UI for uploading, cropping, and searching)
│
├── precompute_embeddings.py       # Computes embeddings, clusters images, trains classifier, and saves artifacts
│
├── requirements.txt               # Lists all required Python packages
```

---

## 2. Basic Setup and Running Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mykytavarenyk/visual-product-search.git
   cd visual-product-search
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Precompute Artifacts:**

   Generate embeddings, cluster assignments, and train the classifier by running:

   ```bash
   python precompute_embeddings.py
   ```

   This script saves the artifacts in the `data/artifacts/` directory.

4. **Launch the Streamlit App:**


   ```bash
   streamlit run app.py
   ```

---

## 5. Live Demo

A demo video showcasing the complete workflow of the app is available here:

[Live Demo Video](https://drive.google.com/file/d/1_czIuEV6O7wcKNHQ3R6TomD7MJ9ACy3x/view?usp=sharing)
