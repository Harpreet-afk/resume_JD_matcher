from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

sys.path.insert(0, str(ROOT))

from src.embeddings import cache_embeddings, embed_texts, load_model

# Load all raw records
resumes_df = pd.read_csv(DATA_DIR / "raw" / "Resume" / "Resume.csv")
jds_df = pd.read_csv(
    DATA_DIR / "raw" / "postings.csv",
    usecols=["job_id", "title", "description"],
    low_memory=False,
)

# Make the text columns used for embedding
resumes_df["text"] = resumes_df["Resume_str"].fillna("")
jds_df["description"] = jds_df["description"].fillna("")

model = load_model()

resume_embeddings = embed_texts(resumes_df["text"].tolist(), model, batch_size=32)
cache_embeddings(resume_embeddings, str(DATA_DIR / "resume_embeddings.npy"))

jd_embeddings = embed_texts(jds_df["description"].tolist(), model, batch_size=32)
cache_embeddings(jd_embeddings, str(DATA_DIR / "jd_embeddings.npy"))