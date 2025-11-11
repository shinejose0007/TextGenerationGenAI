# service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch, os, time
from prometheus_client import Counter, Gauge, start_http_server
import numpy as np

# Optional: sentence-transformers for drift embeddings
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_distances
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# ---------------- Configuration ----------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "training", "outputs", "t5-small-experiment")
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
EMBED_BASE = os.path.join(BASE_DIR, "training", "embeddings_baseline.npy")
DRIFT_THRESHOLD_FILE = os.path.join(os.path.dirname(__file__), "..", "monitoring", "drift_threshold.npy")
# Prometheus metrics port (exposes /metrics)
PROM_PORT = 8001

# ---------------- App & Metrics ----------------
app = FastAPI(title="SmartCity Text Gen with Drift Endpoint")

# start prometheus metrics server (runs in same process; fine for PoC)
start_http_server(PROM_PORT)
REQ_COUNT = Counter('inference_requests_total', 'Total inference requests')
LATENCY = Gauge('inference_latency_seconds', 'Inference latency in seconds')
DRIFT_GAUGE = Gauge('model_drift_score', 'Current model drift score (mean cosine distance)')

# ---------------- Model placeholders ----------------
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# baseline embeddings and threshold
baseline_embeddings = None
drift_mean = None
drift_std = None
drift_threshold = None
sbert = None

# ---------------- Request models ----------------
class Req(BaseModel):
    text: str
    max_length: int = 128

class DriftReq(BaseModel):
    texts: list[str]

# ---------------- Utility functions ----------------
def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        model.to(device)

def load_baseline_and_threshold():
    global baseline_embeddings, drift_mean, drift_std, drift_threshold, sbert
    # load baseline embeddings if present
    if os.path.exists(EMBED_BASE):
        try:
            baseline_embeddings = np.load(EMBED_BASE)
        except Exception:
            baseline_embeddings = None
    # load computed threshold (saved by monitoring/compute_threshold.py)
    if os.path.exists(DRIFT_THRESHOLD_FILE):
        try:
            arr = np.load(DRIFT_THRESHOLD_FILE)
            # file saved as [mean, std, threshold]
            if len(arr) >= 3:
                drift_mean, drift_std, drift_threshold = float(arr[0]), float(arr[1]), float(arr[2])
            else:
                drift_threshold = float(arr[-1])
        except Exception:
            drift_threshold = None
    # load sentence-transformer if available
    if SBERT_AVAILABLE:
        try:
            sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        except Exception:
            sbert = None

def compute_drift_score(new_texts: list[str]) -> float:
    """
    Compute mean cosine distance between new_texts embeddings and baseline embeddings.
    Returns a float (mean distance). Raises HTTPException if prerequisites missing.
    """
    if baseline_embeddings is None:
        raise HTTPException(status_code=400, detail="Baseline embeddings not found. Run monitoring/drift.py first.")
    if not SBERT_AVAILABLE:
        raise HTTPException(status_code=500, detail="sentence-transformers not installed on server.")
    if sbert is None:
        raise HTTPException(status_code=500, detail="Failed to load sentence-transformer model.")
    if not new_texts:
        raise HTTPException(status_code=400, detail="No texts provided for drift check.")
    new_embs = sbert.encode(new_texts, show_progress_bar=False)
    dists = cosine_distances(new_embs, baseline_embeddings)
    return float(dists.mean())

# ---------------- FastAPI lifecycle ----------------
@app.on_event("startup")
def startup_event():
    # load inference model and baseline/thresh at startup
    try:
        load_model()
    except Exception as e:
        # don't crash service if model missing, but log (for PoC we want visibility)
        print("Warning: could not load T5 model at startup:", str(e))
    load_baseline_and_threshold()
    print("Service started. Baseline loaded:", baseline_embeddings is not None, "Threshold:", drift_threshold)

# ---------------- Endpoints ----------------
@app.post("/generate")
def generate(req: Req):
    REQ_COUNT.inc()
    start = time.time()
    if not req.text:
        raise HTTPException(status_code=400, detail="text required")
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    input_text = "summarize: " + req.text
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(**tokens, max_length=req.max_length, num_beams=4)
    pred = tokenizer.decode(out[0], skip_special_tokens=True)
    latency = time.time() - start
    LATENCY.set(latency)
    return {"generated": pred, "latency": latency}

@app.post("/drift")
def drift_check(req: DriftReq):
    """
    Accepts JSON { "texts": ["text1", "text2", ...] } and returns:
    { "drift_score": float, "threshold": float|null, "status": "OK"|"ALERT" }
    """
    try:
        score = compute_drift_score(req.texts)
    except HTTPException as e:
        raise e
    DRIFT_GAUGE.set(score)
    status = "OK"
    if drift_threshold is not None:
        status = "ALERT" if score > drift_threshold else "OK"
    return {"drift_score": score, "threshold": drift_threshold, "status": status}
