import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

MODEL = "paraphrase-MiniLM-L6-v2"
EMBED_OUT = os.path.join(os.path.dirname(__file__), "..", "training", "embeddings_baseline.npy")
DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "train_data.csv")

def compute_baseline_embeddings():
    df = pd.read_csv(DATA_CSV).dropna(subset=['text'])
    model = SentenceTransformer(MODEL)
    texts = df['text'].sample(n=min(500, len(df)), random_state=42).tolist()
    embs = model.encode(texts, show_progress_bar=True)
    np.save(EMBED_OUT, embs)
    print("Saved baseline embeddings to", EMBED_OUT)

def detect_drift(new_texts):
    model = SentenceTransformer(MODEL)
    baseline = np.load(EMBED_OUT)
    new_embs = model.encode(new_texts, show_progress_bar=False)
    dists = cosine_distances(new_embs, baseline)
    avg = dists.mean()
    return float(avg)

if __name__ == "__main__":
    compute_baseline_embeddings()
    # example
    print("Drift:", detect_drift(["Test text for drift check"]))