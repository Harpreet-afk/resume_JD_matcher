import numpy as np

import src.retrieval as retrieval
from src.retrieval import build_index, search, search_jds_by_resume, search_resumes_by_jd


class FakeModel:
    def __init__(self, vectors):
        self._vectors = np.array(vectors, dtype=float)

    def encode(self, texts, batch_size=None, show_progress_bar=False, normalize_embeddings=True):
        if len(texts) != len(self._vectors):
            raise ValueError("FakeModel expects exactly one vector per text")
        if normalize_embeddings:
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            return self._vectors / np.where(norms == 0.0, 1.0, norms)
        return self._vectors


def test_search_matrix_matches_top_result():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    query = np.array([[1.0, 0.0]])

    scores, indices = search(corpus, query, top_k=2)

    assert scores.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert indices[0, 0] == 0
    assert scores[0, 0] >= scores[0, 1]


def test_search_index_matches_matrix_search():
    if retrieval.faiss is None:
        return

    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    query = np.array([[1.0, 0.0]])
    index = build_index(corpus)

    direct_scores, direct_indices = search(corpus, query, top_k=2)
    index_scores, index_indices = search(index, query, top_k=2)

    assert np.allclose(direct_scores, index_scores, atol=1e-6)
    assert np.array_equal(direct_indices, index_indices)


def test_bidirectional_search_wrappers():
    resume_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    jd_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    
    resume_embeddings = resume_embeddings / np.linalg.norm(resume_embeddings, axis=1, keepdims=True)
    jd_embeddings = jd_embeddings / np.linalg.norm(jd_embeddings, axis=1, keepdims=True)

    jd_model = FakeModel([[1.0, 0.0]])
    resume_model = FakeModel([[0.0, 1.0]])

    jd_scores, jd_indices = search_resumes_by_jd(["dummy"], resume_embeddings, jd_model, top_k=1)
    resume_scores, resume_indices = search_jds_by_resume(["dummy"], jd_embeddings, resume_model, top_k=1)

    assert jd_indices[0, 0] == 0
    assert resume_indices[0, 0] == 1
    assert jd_scores[0, 0] > 0
    assert resume_scores[0, 0] > 0
