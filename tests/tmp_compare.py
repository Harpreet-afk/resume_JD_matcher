import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

from src.config import get_broad_field, infer_broad_field_from_text, synthesize_labels
from src.retrieval import search

repo_root = Path(r'C:\Users\hp\Documents\resume_JD_matcher')
resumes_df = pd.read_csv(repo_root/'data'/'raw'/'Resume'/'Resume.csv', encoding='utf-8')
postings_df = pd.read_csv(repo_root/'data'/'raw'/'postings.csv', encoding='utf-8')
resume_embeddings = np.load(repo_root/'data'/'resume_embeddings.npy')
jd_embeddings = np.load(repo_root/'data'/'jd_embeddings.npy')

resumes_df = resumes_df.copy()
resumes_df['category'] = resumes_df['Category'].fillna('other')
resumes_df['broad_field'] = resumes_df['category'].apply(get_broad_field)
resumes_df['id'] = resumes_df['ID'].astype(str)

postings_df = postings_df.copy()
jds_df = postings_df[['job_id','title','description']].copy()
jds_df = jds_df.rename(columns={'job_id': 'id'})
jds_df['title'] = jds_df['title'].fillna('')
jds_df['description'] = jds_df['description'].fillna('')
jds_df['text'] = jds_df['title'] + ' ' + jds_df['description']
jds_df['category'] = jds_df['text'].apply(infer_broad_field_from_text)
jds_df['broad_field'] = jds_df['category'].apply(get_broad_field)

n_jds = 1000
jds_df = jds_df.head(n_jds).copy()
jd_embeddings = jd_embeddings[:n_jds]

labels_df = synthesize_labels(resumes_df, jds_df, sample_negatives=5)
label_dict = {(str(row.resume_id), str(row.jd_id)): int(row.label) for row in labels_df.itertuples(index=False)}

scores, indices = search(resume_embeddings, jd_embeddings, top_k=25)

rows = []
for jd_idx, jd_id in enumerate(jds_df['id'].astype(str)):
    jd_row = jds_df.iloc[jd_idx]
    jd_text = ' '.join([str(v) for v in [jd_row['title'], jd_row['description']] if pd.notna(v)])
    for rank, resume_idx in enumerate(indices[jd_idx], start=1):
        resume_row = resumes_df.iloc[resume_idx]
        resume_id = str(resume_row['id'])
        label = label_dict.get((resume_id, jd_id), 0)
        rows.append({
            'jd_id': jd_id,
            'resume_id': resume_id,
            'rank': rank,
            'score': float(scores[jd_idx, rank-1]),
            'reciprocal_rank': 1.0/rank,
            'score_gap_to_top': float(scores[jd_idx, rank-1] - scores[jd_idx, 0]),
            'same_category': int(resume_row['category'] == jd_row['category']),
            'same_broad_field': int(resume_row['broad_field'] == jd_row['broad_field'] and resume_row['category'] != jd_row['category']),
            'resume_text_length': len(str(resume_row['Resume_str'] or '')),
            'jd_text_length': len(jd_text),
            'token_overlap': len(set(str(resume_row['Resume_str'] or '').lower().split()) & set(jd_text.lower().split())) / max(1, len(set(str(resume_row['Resume_str'] or '').lower().split()) | set(jd_text.lower().split()))),
            'label': label,
        })

feature_frame = pd.DataFrame(rows)

# baseline metrics

def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    positions = np.arange(1, relevances.size+1, dtype=float)
    return float(np.sum((2**relevances - 1)/np.log2(positions+1)))

def ndcg_at_k(relevances, k):
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return float(actual/ideal) if ideal > 0 else 0.0

def first_rel_mrr(relevances):
    for i, r in enumerate(relevances, start=1):
        if r > 0:
            return 1.0/i
    return 0.0

base_ndcg5=[]
base_ndcg10=[]
base_mrr=[]
base_p5=[]
for _, group in feature_frame.groupby('jd_id'):
    labels = group.sort_values('rank')['label'].tolist()
    base_ndcg5.append(ndcg_at_k(labels, 5))
    base_ndcg10.append(ndcg_at_k(labels, 10))
    base_mrr.append(first_rel_mrr(labels))
    base_p5.append(sum(1 for x in labels[:5] if x > 0) / 5.0)

# reranker training
feature_cols = ['rank','score','reciprocal_rank','score_gap_to_top','same_category','same_broad_field','resume_text_length','jd_text_length','token_overlap']
X = feature_frame[feature_cols]
y = feature_frame['label']
groups = feature_frame['jd_id']

kf = GroupKFold(n_splits=3)
reranker_ndcg5=[]
reranker_ndcg10=[]
reranker_mrr=[]
reranker_p5=[]
for train_idx, valid_idx in kf.split(X, y, groups):
    model = LGBMRegressor(objective='regression', n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    valid = feature_frame.iloc[valid_idx].copy()
    valid['predicted_score'] = model.predict(X.iloc[valid_idx])
    for _, group in valid.groupby('jd_id'):
        labels = group.sort_values('predicted_score', ascending=False)['label'].tolist()
        reranker_ndcg5.append(ndcg_at_k(labels, 5))
        reranker_ndcg10.append(ndcg_at_k(labels, 10))
        reranker_mrr.append(first_rel_mrr(labels))
        reranker_p5.append(sum(1 for x in labels[:5] if x > 0) / 5.0)

print('subset_jds', n_jds)
print('baseline_nDCG@5', round(float(np.mean(base_ndcg5)), 4))
print('baseline_nDCG@10', round(float(np.mean(base_ndcg10)), 4))
print('baseline_MRR', round(float(np.mean(base_mrr)), 4))
print('baseline_P@5', round(float(np.mean(base_p5)), 4))
print('reranker_nDCG@5', round(float(np.mean(reranker_ndcg5)), 4))
print('reranker_nDCG@10', round(float(np.mean(reranker_ndcg10)), 4))
print('reranker_MRR', round(float(np.mean(reranker_mrr)), 4))
print('reranker_P@5', round(float(np.mean(reranker_p5)), 4))
