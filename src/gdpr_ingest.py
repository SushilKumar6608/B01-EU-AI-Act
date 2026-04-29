#dpr_ingest.py
import os
import pickle
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/gdpr.pdf"
INDEX_PATH = "data/gdpr_faiss_index.bin"
CHUNKS_PATH = "data/gdpr_chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_by_article(text):
    import re
    pattern = r'(?=Article\s+\d+[\s\n])'
    parts = re.split(pattern, text)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) > 100:
            chunks.append(part)
    return chunks


def build_and_save():
    print("Step 1: Extracting GDPR text...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(text):,} characters.")

    print("Step 2: Chunking by article...")
    chunks = chunk_by_article(text)
    print(f"Created {len(chunks)} chunks.")

    print("Step 3: Embedding...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True)

    print("Step 4: Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    print("Step 5: Saving...")
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done. {len(chunks)} GDPR chunks indexed.")


if __name__ == "__main__":
    build_and_save()