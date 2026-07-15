import pandas as pd

CATEGORY_GROUPS = {
    "tech": ["Information-Technology", "Data-Science", "Web-Developing",
              "Automation-Testing", "Database", "DevOps-Engineer",
              "Network-Security-Engineer", "Blockchain", "ETL-Developer"],
    "engineering": ["Mechanical-Engineer", "Electrical-Engineering",
                     "Civil-Engineer", "Engineering"],
    "business": ["Business-Development", "Sales", "Digital-Media",
                  "Operations-Manager", "PMO"],
    "finance": ["Accountant", "Finance", "Banking"],
    "health": ["Health-and-fitness", "Healthcare"],
    "creative": ["Arts", "Designer", "Apparel"],
    "hr": ["HR", "Advocate", "Public-Relations"],
    "education": ["Teacher"],
}

CATEGORY_KEYWORDS = {
    "tech": ["software", "developer", "programming", "data", "machine learning", "ai", "cloud", "database", "security", "web", "devops", "testing", "automation", "it", "technical"],
    "engineering": ["engineer", "engineering", "mechanical", "electrical", "civil", "architect", "construction", "manufacturing", "structural"],
    "business": ["sales", "business", "operations", "product", "marketing", "manager", "project", "strategy", "planning"],
    "finance": ["finance", "accountant", "banking", "accounting", "audit", "treasury", "investment"],
    "health": ["health", "medical", "hospital", "therapy", "nurse", "care", "fitness", "wellness"],
    "creative": ["design", "artist", "creative", "media", "content", "writer", "editor", "marketing"],
    "hr": ["hr", "human resources", "recruiter", "talent", "people", "advocate"],
    "education": ["teacher", "education", "trainer", "instructor", "curriculum"],
}


def get_broad_field(category: str) -> str:
    """Map a specific category to its broad field."""
    if not category:
        return "other"

    category_norm = str(category).strip().lower()
    if category_norm in CATEGORY_GROUPS:
        return category_norm

    for field, categories in CATEGORY_GROUPS.items():
        for candidate in categories:
            if category_norm == str(candidate).strip().lower():
                return field
    return "other"


def infer_broad_field_from_text(text: str) -> str:
    """Infer a broad field from free-form text using keyword matching."""
    if not text:
        return "other"

    text_norm = str(text).strip().lower()
    for field, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_norm for keyword in keywords):
            return field
    return "other"


def _resolve_id_column(df, fallback_name: str) -> pd.Series:
    if "id" in df.columns:
        return df["id"]
    if fallback_name in df.columns:
        return df[fallback_name]
    if "ID" in df.columns:
        return df["ID"]
    return df.index.to_series()


def _resolve_category_column(df, fallback_columns=None) -> pd.Series:
    fallback_columns = fallback_columns or ["category", "Category", "cat"]
    for column in fallback_columns:
        if column in df.columns:
            return df[column]
    return pd.Series([None] * len(df), index=df.index)


def synthesize_labels(resumes_df, jds_df, sample_negatives=5):
    """Create training labels for the re-ranker.

    For each JD, pair it with:
    - Label 2: resumes from same category
    - Label 1: resumes from same broad field
    - Label 0: random resumes from different fields (sampled)
    """
    resumes_df = resumes_df.copy()
    jds_df = jds_df.copy()

    if "id" not in resumes_df.columns:
        resumes_df["id"] = _resolve_id_column(resumes_df, "resume_id")
    if "id" not in jds_df.columns:
        jds_df["id"] = _resolve_id_column(jds_df, "job_id")

    if "category" not in resumes_df.columns:
        resumes_df["category"] = _resolve_category_column(resumes_df, ["Category", "category"])
    if "broad_field" not in resumes_df.columns:
        resumes_df["broad_field"] = resumes_df["category"].apply(get_broad_field)

    if "category" not in jds_df.columns:
        jd_text = jds_df[[col for col in ["title", "description", "text"] if col in jds_df.columns]].fillna("").astype(str).agg(lambda s: " ".join(s), axis=1)
        jds_df["category"] = jd_text.apply(infer_broad_field_from_text)
    if "broad_field" not in jds_df.columns:
        jds_df["broad_field"] = jds_df["category"].apply(get_broad_field)

    pairs = []
    for _, jd in jds_df.iterrows():
        jd_cat = jd["category"]
        jd_field = jd["broad_field"] if "broad_field" in jd else get_broad_field(jd_cat)

        exact = resumes_df[resumes_df["category"] == jd_cat]
        for _, res in exact.iterrows():
            pairs.append((res["id"], jd["id"], 2))

        field_match = resumes_df[
            (resumes_df["category"] != jd_cat) &
            (resumes_df["broad_field"] == jd_field)
        ]
        for _, res in field_match.iterrows():
            pairs.append((res["id"], jd["id"], 1))

        irrelevant = resumes_df[resumes_df["broad_field"] != jd_field]
        sampled = irrelevant.sample(min(sample_negatives, len(irrelevant)))
        for _, res in sampled.iterrows():
            pairs.append((res["id"], jd["id"], 0))

    return pd.DataFrame(pairs, columns=["resume_id", "jd_id", "label"])
