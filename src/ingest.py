#ingest.py
import os
import pickle
import fitz  # pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/eu_ai_act.pdf"
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"
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
    # Split on "Article X" boundaries — the natural legal unit
    pattern = r'(?=Article\s+\d+[\s\n])'
    parts = re.split(pattern, text)
    
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) > 100:  # skip empty or near-empty splits
            chunks.append(part)
    
    return chunks


def embed_chunks(chunks, model_name):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(chunks)} chunks — this runs locally, no API cost...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_artifacts(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"FAISS index saved to {INDEX_PATH}")
    print(f"Chunks saved to {CHUNKS_PATH}")


def main():
    print("Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(text):,} characters from PDF.")

    print("\nStep 2: Chunking by article...")
    chunks = chunk_by_article(text)
    print(f"Created {len(chunks)} chunks.")

    print("\nStep 3: Embedding chunks...")
    embeddings, _ = embed_chunks(chunks, EMBED_MODEL)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nStep 4: Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"Index contains {index.ntotal} vectors.")

    print("\nStep 5: Saving artifacts to disk...")
    save_artifacts(index, chunks)

    print("\nDone. Ingest complete.")
    print(f"Total chunks indexed: {len(chunks)}")
    print("Sample chunk preview:")
    print("-" * 40)
    print(chunks[0][:300])


if __name__ == "__main__":
    main()