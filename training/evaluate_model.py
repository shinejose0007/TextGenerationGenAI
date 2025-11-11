import os
import pandas as pd
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from evaluate import load  # updated import

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "outputs", "t5-small-experiment")
DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "train_data.csv")
EVAL_CSV = os.path.join(os.path.dirname(__file__), "eval_results.csv")

# Device setup (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model

def generate_and_score(batch_size=8):
    tokenizer, model = load_model()

    # Load and sample data
    df = pd.read_csv(DATA_CSV).dropna(subset=['text', 'summary'])
    df = df.sample(n=min(50, len(df)), random_state=42)
    inputs = ["summarize: " + t for t in df['text'].tolist()]
    refs = df['summary'].tolist()

    preds = []

    # Batch generation for efficiency
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        encodings = tokenizer(batch_inputs, return_tensors="pt", truncation=True,
                              padding=True, max_length=512).to(DEVICE)
        outputs = model.generate(**encodings, max_length=128, num_beams=4)
        batch_preds = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        preds.extend(batch_preds)

    # Load metric
    sacrebleu = load("sacrebleu")
    bleu = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])

    # Save results
    eval_df = pd.DataFrame({"input": inputs, "pred": preds, "ref": refs})
    eval_df.to_csv(EVAL_CSV, index=False)

    print("BLEU score:", bleu)
    print("Saved evaluation results to", EVAL_CSV)

if __name__ == "__main__":
    generate_and_score()
