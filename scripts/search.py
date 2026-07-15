from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.embeddings import load_model, load_cached_embeddings
from src.retrieval import build_index, search_jds_by_resume, search_resumes_by_jd


def load_metadata(csv_path: Path, expected_columns):
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path, usecols=expected_columns, low_memory=False)
    except Exception:
        return None


def format_match_rows(indices, scores, metadata, kind):
    rows = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        meta_str = ""
        if metadata is not None and idx < len(metadata):
            row = metadata.iloc[idx]
            if kind == "resume":
                meta_str = f"ID={row.get('ID', idx)} Category={row.get('Category', '')}"
            else:
                meta_str = f"job_id={row.get('job_id', idx)} title={row.get('title', '')}"
        rows.append(f"{rank}. index={idx} score={score:.4f} {meta_str}")
    return rows


def parse_args():
    parser = ArgumentParser(description="Bidirectional resume / JD similarity search")
    parser.add_argument(
        "--direction",
        choices=["jd2resume", "resume2jd"],
        required=True,
        help="Search direction: jd2resume finds resumes for a JD; resume2jd finds JDs for a resume.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text to search against the target corpus.",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Path to a text file containing the query.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top matches to return.")
    parser.add_argument(
        "--use-index",
        action="store_true",
        help="Build and use a FAISS index for faster retrieval.",
    )
    parser.add_argument(
        "--resume-embeddings",
        type=Path,
        default=DATA_DIR / "resume_embeddings.npy",
        help="Path to cached resume embeddings.",
    )
    parser.add_argument(
        "--jd-embeddings",
        type=Path,
        default=DATA_DIR / "jd_embeddings.npy",
        help="Path to cached JD embeddings.",
    )
    parser.add_argument(
        "--resumes-csv",
        type=Path,
        default=DATA_DIR / "raw" / "Resume" / "Resume.csv",
        help="Optional PATH to resume metadata CSV for result labels.",
    )
    parser.add_argument(
        "--jds-csv",
        type=Path,
        default=DATA_DIR / "raw" / "postings.csv",
        help="Optional PATH to JD metadata CSV for result labels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.query_file and args.query:
        raise ValueError("Specify only one of --query or --query-file.")

    if args.query_file:
        query_text = args.query_file.read_text(encoding="utf-8").strip()
    elif args.query:
        query_text = args.query.strip()
    else:
        raise ValueError("Please provide either --query or --query-file.")

    if not query_text:
        raise ValueError("The query text is empty.")

    model = load_model()
    resume_embeddings = load_cached_embeddings(str(args.resume_embeddings))
    jd_embeddings = load_cached_embeddings(str(args.jd_embeddings))

    resume_metadata = load_metadata(args.resumes_csv, ["ID", "Category"])
    jd_metadata = load_metadata(args.jds_csv, ["job_id", "title"])

    if args.direction == "jd2resume":
        index = None
        if args.use_index:
            try:
                index = build_index(resume_embeddings)
            except ImportError as exc:
                raise SystemExit(str(exc)) from exc

        scores, indices = search_resumes_by_jd([query_text], resume_embeddings, model, args.top_k, index)
        rows = format_match_rows(indices, scores, resume_metadata, kind="resume")
        heading = "Top resumes for the given JD"
    else:
        index = None
        if args.use_index:
            try:
                index = build_index(jd_embeddings)
            except ImportError as exc:
                raise SystemExit(str(exc)) from exc

        scores, indices = search_jds_by_resume([query_text], jd_embeddings, model, args.top_k, index)
        rows = format_match_rows(indices, scores, jd_metadata, kind="jd")
        heading = "Top JDs for the given resume"

    print(heading)
    print("=" * len(heading))
    print(query_text)
    print("")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
