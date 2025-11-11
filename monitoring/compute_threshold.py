# monitoring/compute_threshold.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import os

MODEL = "paraphrase-MiniLM-L6-v2"
EMBED_BASE = os.path.join(os.path.dirname(__file__), "..", "training", "embeddings_baseline.npy")
DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "train_data.csv")

def load_baseline():
    return np.load(EMBED_BASE)

def batch_score(new_texts, model):
    new_embs = model.encode(new_texts, show_progress_bar=False)
    baseline = load_baseline()
    dists = cosine_distances(new_embs, baseline)
    return dists.mean()

if __name__ == "__main__":
    model = SentenceTransformer(MODEL)
    df = pd.read_csv(DATA_CSV).dropna(subset=['text'])
    # sample several batches from historical data to estimate distribution
    scores = []
    for i in range(30):
        sample = df['text'].sample(n=min(20, len(df)), random_state=i).tolist()
        scores.append(batch_score(sample, model))
    scores = np.array(scores)
    print("mean:", scores.mean(), "std:", scores.std())
    # threshold: mean + 3*std (adjust k as needed)
    thr = scores.mean() + 3 * scores.std()
    print("suggested_threshold:", thr)
    np.save(os.path.join(os.path.dirname(__file__), "drift_threshold.npy"), np.array([scores.mean(), scores.std(), thr]))
    print("Saved drift_threshold.npy")
