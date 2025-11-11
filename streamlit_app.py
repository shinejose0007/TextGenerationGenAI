import streamlit as st
import time, os, numpy as np, pandas as pd
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
from prometheus_client import start_http_server, Counter, Gauge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

MODEL_DIR = os.getenv("MODEL_DIR", "training/outputs/t5-small-experiment")
EMBED_BASELINE = os.getenv("EMBED_BASELINE", "training/embeddings_baseline.npy")

# Start Prometheus metrics server (only once)
start_http_server(8001)

# Use st.cache_resource to prevent duplicate registration
@st.cache_resource
def get_metrics():
    inf_count = Counter("streamlit_inference_total", "Total Streamlit inference calls")
    lat = Gauge("streamlit_last_latency_seconds", "Last inference latency")
    return inf_count, lat

INF_COUNT, LAT = get_metrics()

# Load model and SBERT embeddings
@st.cache_resource
def load_model():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_resource
def load_sbert(model_name="paraphrase-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Streamlit UI
st.set_page_config(page_title="SmartCity Gen Demo", layout="wide")
st.title("SmartCity — Generative AI Demo")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Input / Generation")
    input_text = st.text_area("Enter text (incident / report)", height=160)
    max_len = st.slider("Max tokens", 32, 256, 128)
    gen_btn = st.button("Generate summary")

with col2:
    st.subheader("Model & Metrics")
    st.write("Model dir:", MODEL_DIR)
    st.write("Prometheus metrics port: 8001")

tokenizer, model, device = load_model()

if gen_btn and input_text.strip():
    INF_COUNT.inc()  # Increment metric
    t0 = time.time()
    prefix = "summarize: " + input_text
    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(**enc, max_length=max_len, num_beams=4)
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    latency = time.time() - t0
    LAT.set(latency)  # Update metric
    st.subheader("Generated")
    st.info(generated)
    st.write("Latency: {:.3f}s".format(latency))
    log_df = pd.DataFrame([dict(input=input_text, generated=generated, latency=latency, ts=time.time())])
    st.download_button("Download sample result (CSV)", log_df.to_csv(index=False), file_name="sample_result.csv")

st.markdown("---")
st.subheader("Lightweight Drift Check (Demo)")
sbert = load_sbert()
if os.path.exists(EMBED_BASELINE):
    baseline = np.load(EMBED_BASELINE)
    sample_text = st.text_area("Optional: Paste a small batch of recent texts (one per line)", height=120)
    if st.button("Check drift") and sample_text.strip():
        new_texts = [t.strip() for t in sample_text.splitlines() if t.strip()]
        new_embs = sbert.encode(new_texts, show_progress_bar=False)
        dists = cosine_distances(new_embs, baseline)
        score = float(dists.mean())
        st.metric("Drift score (mean cosine dist)", f"{score:.4f}")
        st.write("Interpretation: larger = more drift vs. baseline. Use thresholds from eval.")
else:
    st.warning("Baseline embeddings not found. Run `python monitoring/drift.py` first.")
