#retriever.py
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

_index = None
_chunks = None
_model = None


def load_artifacts():
    global _index, _chunks, _model
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _chunks is None:
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)


def retrieve(query, top_k=5):
    load_artifacts()
    query_embedding = _model.encode([query]).astype(np.float32)
    distances, indices = _index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "chunk": _chunks[idx],
                "distance": float(distances[0][i]),
                "index": int(idx)
            })
    return results


if __name__ == "__main__":
    # Quick test — costs zero API credits
    test_query = "facial recognition system used by police"
    print(f"Test query: {test_query}\n")
    results = retrieve(test_query, top_k=3)
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (distance: {r['distance']:.4f}) ---")
        print(r["chunk"][:400])
        print()