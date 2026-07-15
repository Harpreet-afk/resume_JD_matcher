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

def get_broad_field(category: str) -> str:
    """Map a specific category to its broad field."""
    for field, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return field
    return "other"

def synthesize_labels(resumes_df, jds_df, sample_negatives=5):
    """Create training labels for the re-ranker.

    For each JD, pair it with:
    - Label 2: resumes from same category
    - Label 1: resumes from same broad field
    - Label 0: random resumes from different fields (sampled)
    """
    pairs = []
    for _, jd in jds_df.iterrows():
        jd_cat = jd["category"]
        jd_field = get_broad_field(jd_cat)

        # Label 2: exact category match
        exact = resumes_df[resumes_df["category"] == jd_cat]
        for _, res in exact.iterrows():
            pairs.append((res["id"], jd["id"], 2))

        # Label 1: same broad field, different specific category
        field_match = resumes_df[
            (resumes_df["category"] != jd_cat) &
            (resumes_df["broad_field"] == jd_field)
        ]
        for _, res in field_match.iterrows():
            pairs.append((res["id"], jd["id"], 1))

        # Label 0: different field (sampled to avoid explosion)
        irrelevant = resumes_df[resumes_df["broad_field"] != jd_field]
        sampled = irrelevant.sample(min(sample_negatives, len(irrelevant)))
        for _, res in sampled.iterrows():
            pairs.append((res["id"], jd["id"], 0))

    return pd.DataFrame(pairs, columns=["resume_id", "jd_id", "label"])