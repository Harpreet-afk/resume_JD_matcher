
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from src.embeddings import load_model, embed_texts
from src.retrieval import load_index, search

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESUME_CSV = DATA_DIR / "raw" / "Resume" / "Resume.csv"
RESUME_EMBED_PATH = DATA_DIR / "resume_embeddings.npy"
RESUME_INDEX_PATH = DATA_DIR / "resume.index"


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower()


def _token_overlap(query: str, text: str) -> float:
    query_tokens = set(_normalize_text(query).split())
    text_tokens = set(_normalize_text(text).split())
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens | text_tokens)


def _load_resume_data() -> pd.DataFrame:
    resumes = pd.read_csv(RESUME_CSV, encoding="utf-8")
    if "ID" in resumes.columns:
        resumes["id"] = resumes["ID"].astype(str)
    elif "id" in resumes.columns:
        resumes["id"] = resumes["id"].astype(str)
    else:
        resumes["id"] = resumes.index.astype(str)

    resumes["preview"] = resumes.get("Resume_str", "").fillna("").astype(str).str.replace("\n", " ").str[:600]
    resumes["category"] = resumes.get("Category", "").fillna("other")
    return resumes.reset_index(drop=True)


def _rerank_candidates(scores: np.ndarray, indices: np.ndarray, query_text: str, resume_texts: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    reranked = []
    for score, idx in zip(scores.flatten(), indices.flatten()):
        resume_text = resume_texts[idx][0]
        overlap = _token_overlap(query_text, resume_text)
        combined = float(score) * 0.7 + overlap * 0.3
        reranked.append((combined, score, idx))

    reranked.sort(key=lambda row: row[0], reverse=True)
    new_indices = np.array([[row[2] for row in reranked]], dtype=np.int64)
    new_scores = np.array([[row[1] for row in reranked]], dtype=float)
    return new_scores, new_indices


st.set_page_config(page_title="Resume-JD Matcher", layout="wide")
st.title("Resume-JD Matcher")
st.markdown("Paste a job description to find the best-matching resumes.")

resume_df = _load_resume_data()

with st.sidebar:
    top_k = st.slider("Number of results", 5, 50, 10)
    use_reranker = st.checkbox("Enable heuristic reranker", value=False)
    search_mode = st.selectbox("Search source", ["Auto", "FAISS index", "Raw resume embeddings"])
    show_preview = st.checkbox("Show resume preview", value=True)

jd_text = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste a job description here...",
)

if st.button("🔍 Find Matching Resumes"):
    if not jd_text:
        st.warning("Please enter a job description before searching.")
    else:
        with st.spinner("Embedding and searching..."):
            model = load_model()
            jd_emb = embed_texts([jd_text], model)
            resume_texts = resume_df["Resume_str"].fillna("").astype(str).tolist()
            index = None
            source_label = "raw resume embeddings"
            if search_mode in {"Auto", "FAISS index"}:
                try:
                    index = load_index(str(RESUME_INDEX_PATH))
                    source_label = "FAISS index"
                except Exception:
                    if search_mode == "FAISS index":
                        st.warning("FAISS index unavailable; falling back to raw resume embeddings.")
                    index = np.load(RESUME_EMBED_PATH)
            else:
                index = np.load(RESUME_EMBED_PATH)

            scores, indices = search(index, jd_emb, top_k=top_k)
            if use_reranker:
                scores, indices = _rerank_candidates(
                    scores,
                    indices,
                    jd_text,
                    list(zip(resume_texts, resume_df["category"].tolist())),
                )

        st.success(f"Search completed using {source_label}.")
        results = []
        for rank, (score, resume_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            result_row = resume_df.iloc[resume_idx]
            results.append(
                {
                    "Rank": rank,
                    "Resume ID": result_row["id"],
                    "Score": round(float(score), 4),
                    "Category": result_row.get("category", ""),
                    "Preview": result_row["preview"] if show_preview else "",
                }
            )

        st.write(f"### Top {len(results)} matches")
        st.table(pd.DataFrame(results))

        if show_preview:
            for item in results:
                with st.expander(f"Resume {item['Resume ID']} — Score {item['Score']}"):
                    st.markdown(f"**Category:** {item['Category']}")
                    st.markdown(item["Preview"])
