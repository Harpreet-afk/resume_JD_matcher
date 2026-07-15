import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

repo_root = Path.cwd().resolve()
sys.path.insert(0, str(repo_root))
from src.config import get_broad_field, infer_broad_field_from_text, synthesize_labels
from src.embeddings import load_model
from src.retrieval import search


def _normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()

def _token_overlap(resume_text, jd_text):
    resume_tokens = set(_normalize_text(resume_text).split())
    jd_tokens = set(_normalize_text(jd_text).split())
    if not resume_tokens or not jd_tokens:
        return 0.0
    return len(resume_tokens & jd_tokens) / len(resume_tokens | jd_tokens)

def _ndcg_at_k(labels, k=10):
    labels = np.asarray(labels, dtype=float)[:k]
    if labels.size == 0:
        return 0.0
    gains = np.power(2.0, labels) - 1.0
    discounts = np.log2(np.arange(2, labels.size + 2, dtype=float))
    dcg = np.sum(gains / discounts)
    ideal = np.sort(labels)[::-1]
    ideal_gains = np.power(2.0, ideal) - 1.0
    ideal_discounts = np.log2(np.arange(2, ideal.size + 2, dtype=float))
    ideal_dcg = np.sum(ideal_gains / ideal_discounts)
    return 0.0 if ideal_dcg == 0 else dcg / ideal_dcg

def first_relevant_mrr(labels):
    for i, lab in enumerate(labels, start=1):
        if lab > 0:
            return 1.0 / i
    return 0.0

repo_root = Path.cwd().resolve()

# Load data
resumes_df = pd.read_csv(repo_root / 'data' / 'raw' / 'Resume' / 'Resume.csv', encoding='utf-8')
postings_df = pd.read_csv(repo_root / 'data' / 'raw' / 'postings.csv', encoding='utf-8')
resume_embeddings = np.load(repo_root / 'data' / 'resume_embeddings.npy')
resumes_df = resumes_df.copy()
resumes_df['category'] = resumes_df['Category'].fillna('other')
resumes_df['broad_field'] = resumes_df['category'].apply(get_broad_field)
resumes_df['id'] = resumes_df['ID'].astype(str)
resumes_df = resumes_df.iloc[:resume_embeddings.shape[0]].copy()

jds_df = postings_df[['job_id','title','description']].copy()
jds_df = jds_df.rename(columns={'job_id': 'id'})
jds_df['title'] = jds_df['title'].fillna('')
jds_df['description'] = jds_df['description'].fillna('')
jds_df['text'] = jds_df['title'] + ' ' + jds_df['description']
jds_df['category'] = jds_df['text'].apply(infer_broad_field_from_text)
jds_df['broad_field'] = jds_df['category'].apply(get_broad_field)
jds_df = jds_df.head(200).copy()

# labels and features for reranker
labels_df = synthesize_labels(resumes_df, jds_df, sample_negatives=5)
resume_ids = resumes_df['id'].tolist()
jd_ids = jds_df['id'].tolist()

scores, indices = search(resume_embeddings, jd_embeddings := load_model().encode((jds_df['title'] + ' ' + jds_df['description']).tolist(), batch_size=32, show_progress_bar=False, normalize_embeddings=True), top_k=25)

rows = []
for jd_idx, jd_id in enumerate(jd_ids):
    jd_row = jds_df.iloc[jd_idx]
    jd_text = ' '.join(str(v) for v in [jd_row.get('title',''), jd_row.get('description','')] if pd.notna(v))
    jd_cat = jd_row['category']
    jd_field = jd_row['broad_field']
    for rank, resume_idx in enumerate(indices[jd_idx], start=1):
        resume_row = resumes_df.iloc[resume_idx]
        rows.append({
            'jd_id': str(jd_id),
            'resume_id': str(resume_row['id']),
            'rank': rank,
            'score': float(scores[jd_idx, rank - 1]),
            'reciprocal_rank': 1.0/rank,
            'score_gap_to_top': float(scores[jd_idx, rank - 1] - scores[jd_idx, 0]),
            'same_category': int(resume_row['category'] == jd_cat),
            'same_broad_field': int(resume_row['broad_field'] == jd_field and resume_row['category'] != jd_cat),
            'resume_text_length': len(_normalize_text(str(resume_row['Resume_str']))),
            'jd_text_length': len(_normalize_text(jd_text)),
            'token_overlap': _token_overlap(str(resume_row['Resume_str']), jd_text),
            'label': int(labels_df.loc[(labels_df['resume_id'] == resume_row['id']) & (labels_df['jd_id'] == jd_id), 'label'].iloc[0]) if not labels_df.loc[(labels_df['resume_id'] == resume_row['id']) & (labels_df['jd_id'] == jd_id), 'label'].empty else 0,
        })

feature_frame = pd.DataFrame(rows)
feature_frame['jd_id'] = feature_frame['jd_id'].astype(str)
feature_frame['resume_id'] = feature_frame['resume_id'].astype(str)

# baseline metrics
base_ndcg5 = np.mean([_ndcg_at_k(group.sort_values('rank')['label'].to_numpy(), k=5) for _, group in feature_frame.groupby('jd_id')])
base_ndcg10 = np.mean([_ndcg_at_k(group.sort_values('rank')['label'].to_numpy(), k=10) for _, group in feature_frame.groupby('jd_id')])
base_mrr = np.mean([first_relevant_mrr(group.sort_values('rank')['label'].to_numpy()) for _, group in feature_frame.groupby('jd_id')])
base_p5 = np.mean([np.sum(group.sort_values('rank')['label'].to_numpy()[:5] > 0) / min(5, len(group)) for _, group in feature_frame.groupby('jd_id')])

# reranker training with GroupKFold from feature_frame
X = feature_frame[['rank','score','reciprocal_rank','score_gap_to_top','same_category','same_broad_field','resume_text_length','jd_text_length','token_overlap']]
y = feature_frame['label']
groups = feature_frame['jd_id']

splitter = GroupKFold(n_splits=3)
reranker_ndcg5=[]
reranker_ndcg10=[]
reranker_mrr=[]
reranker_p5=[]
for train_idx, valid_idx in splitter.split(X, y, groups):
    model = LGBMRegressor(objective='regression', n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    valid = feature_frame.iloc[valid_idx].copy()
    valid['predicted_score'] = model.predict(X.iloc[valid_idx])
    for _, group in valid.groupby('jd_id'):
        scores = group.sort_values('predicted_score', ascending=False)['label'].to_numpy()
        reranker_ndcg5.append(_ndcg_at_k(scores, k=5))
        reranker_ndcg10.append(_ndcg_at_k(scores, k=10))
        reranker_mrr.append(first_relevant_mrr(scores))
        reranker_p5.append(np.sum(scores[:5] > 0) / min(5, len(scores)))

print('baseline_nDCG@5', round(base_ndcg5, 4))
print('baseline_nDCG@10', round(base_ndcg10, 4))
print('baseline_MRR', round(base_mrr, 4))
print('baseline_P@5', round(base_p5, 4))
print('reranker_nDCG@5', round(np.mean(reranker_ndcg5), 4))
print('reranker_nDCG@10', round(np.mean(reranker_ndcg10), 4))
print('reranker_MRR', round(np.mean(reranker_mrr), 4))
print('reranker_P@5', round(np.mean(reranker_p5), 4))
