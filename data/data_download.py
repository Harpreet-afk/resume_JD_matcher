import os
import subprocess
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download_resumes():
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "snehaanbhawal/resume-dataset",
        "-p", str(DATA_DIR / "raw"),
        "--unzip"
    ], check=True)
    print("Resume dataset downloaded.")

def download_job_postings():
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "arshkon/linkedin-job-postings",
        "-p", str(DATA_DIR / "raw"),
        "--unzip"
    ], check=True)
    print("Job postings dataset downloaded.")

def filter_jd_dataset():
    raw_path = DATA_DIR / "raw" / "postings.csv"
    df = pd.read_csv(raw_path)

    tech_keywords = [
        'software', 'engineer', 'developer', 'data', 'analyst',
        'python', 'java', 'machine learning', 'devops', 'cloud',
        'IT', 'frontend', 'backend', 'full stack', 'database',
        'network', 'security', 'systems', 'architect', 'QA',
        'test', 'automation', 'AI', 'deep learning', 'NLP'
    ]
    pattern = '|'.join(tech_keywords)

    filtered = df[
        df['title'].str.contains(pattern, case=False, na=False) |
        df['description'].str.contains(pattern, case=False, na=False)
    ].copy()

    # Drop nulls in critical columns
    filtered = filtered.dropna(subset=['title', 'description'])
    # Deduplicate
    filtered = filtered.drop_duplicates(subset=['description'])

    out_path = DATA_DIR / "jd_filtered.parquet"
    filtered.to_parquet(out_path, index=False)
    print(f"Filtered JDs: {len(filtered)} rows → {out_path}")

if __name__ == "__main__":
    download_resumes()
    download_job_postings()
    filter_jd_dataset()