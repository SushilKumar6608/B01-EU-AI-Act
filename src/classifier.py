#classifier.py
import os
import pickle
import numpy as np
import anthropic
import faiss
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"

# Index paths
INDEXES = {
    "EU AI Act": {
        "index": "data/faiss_index.bin",
        "chunks": "data/chunks.pkl",
    },
    "GDPR": {
        "index": "data/gdpr_faiss_index.bin",
        "chunks": "data/gdpr_chunks.pkl",
    },
}

_loaded = {}
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _load_index(regulation):
    global _loaded
    if regulation not in _loaded:
        paths = INDEXES[regulation]
        index = faiss.read_index(paths["index"])
        with open(paths["chunks"], "rb") as f:
            chunks = pickle.load(f)
        _loaded[regulation] = {"index": index, "chunks": chunks}
    return _loaded[regulation]


def retrieve(query, regulation, top_k=5):
    data = _load_index(regulation)
    model = _get_model()
    embedding = model.encode([query]).astype(np.float32)
    distances, indices = data["index"].search(embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "chunk": data["chunks"][idx],
                "distance": float(distances[0][i]),
                "index": int(idx),
            })
    return results


def compute_confidence(results):
    """
    Convert FAISS L2 distances to a confidence score.
    all-MiniLM-L6-v2 typical distance range: 0.3 (excellent) to 1.8 (poor).
    Map to 0-100% accordingly.
    """
    if not results:
        return 0.0, "No relevant articles found"

    # Use best match distance, not average — best match drives relevance
    best_distance = min(r["distance"] for r in results)

    # Scale: 0.3 -> 100%, 1.8 -> 0%
    confidence = max(0.0, min(100.0, (1.8 - best_distance) / (1.8 - 0.3) * 100))

    if confidence >= 50:
        label = "High"
    elif confidence >= 25:
        label = "Medium"
    else:
        label = "Low — retrieved articles may not be directly applicable"

    return round(confidence, 1), label


def extract_text_from_pdf(uploaded_file_bytes):
    """Extract text from uploaded PDF bytes."""
    doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


SYSTEM_PROMPTS = {
    "EU AI Act": """You are an expert in EU AI Act compliance (Regulation 2024/1689).
You will be given a description of an AI system and relevant excerpts from the EU AI Act.

Classify the system into one of four risk tiers:
- UNACCEPTABLE: Prohibited systems under Article 5
- HIGH: High-risk systems under Annex III
- LIMITED: Systems with transparency obligations
- MINIMAL: All other systems

Structure your response exactly as follows:

RISK TIER: [UNACCEPTABLE / HIGH / LIMITED / MINIMAL]

CLASSIFICATION REASONING:
[2-4 sentences citing specific articles]

CITED ARTICLES:
[List each relevant article and one-line summary]

COMPLIANCE CHECKLIST:
[Numbered list of concrete obligations]

IMPORTANT CAVEATS:
[Assumptions made or edge cases to verify with a legal professional]""",

    "GDPR": """You are an expert in GDPR compliance (Regulation 2016/679).
You will be given a description of an AI system and relevant excerpts from GDPR.

Assess the system's GDPR obligations focusing on:
- Lawful basis for processing (Article 6)
- Special category data (Article 9)
- Data subject rights obligations
- Data Protection Impact Assessment (DPIA) requirement (Article 35)
- Controller/processor obligations

Structure your response exactly as follows:

GDPR RISK LEVEL: [HIGH / MEDIUM / LOW]

ASSESSMENT REASONING:
[2-4 sentences citing specific articles]

CITED ARTICLES:
[List each relevant article and one-line summary]

COMPLIANCE CHECKLIST:
[Numbered list of concrete GDPR obligations]

IMPORTANT CAVEATS:
[Assumptions made or edge cases to verify with a legal professional]""",

    "Both": """You are an expert in both EU AI Act (Regulation 2024/1689) and GDPR (Regulation 2016/679) compliance.
You will be given a description of an AI system and relevant excerpts from both regulations.

Provide a combined compliance assessment.

Structure your response exactly as follows:

EU AI ACT RISK TIER: [UNACCEPTABLE / HIGH / LIMITED / MINIMAL]
GDPR RISK LEVEL: [HIGH / MEDIUM / LOW]

EU AI ACT ASSESSMENT:
[2-3 sentences citing specific EU AI Act articles]

GDPR ASSESSMENT:
[2-3 sentences citing specific GDPR articles]

CITED ARTICLES:
[List articles from both regulations]

COMBINED COMPLIANCE CHECKLIST:
[Numbered list covering both EU AI Act and GDPR obligations, clearly labelled]

IMPORTANT CAVEATS:
[Assumptions made or edge cases to verify with a legal professional]""",
}


def classify_system(system_description, regulation="EU AI Act"):
    """Classify a single system description against one or both regulations."""
    if regulation == "Both":
        euaia_results = retrieve(system_description, "EU AI Act", top_k=4)
        gdpr_results = retrieve(system_description, "GDPR", top_k=3)
        all_results = euaia_results + gdpr_results
        euaia_context = "\n\n---\n\n".join([r["chunk"] for r in euaia_results])
        gdpr_context = "\n\n---\n\n".join([r["chunk"] for r in gdpr_results])
        context = f"EU AI ACT EXCERPTS:\n{euaia_context}\n\nGDPR EXCERPTS:\n{gdpr_context}"
        confidence_score, confidence_label = compute_confidence(all_results)
    else:
        results = retrieve(system_description, regulation, top_k=5)
        context = "\n\n---\n\n".join([r["chunk"] for r in results])
        confidence_score, confidence_label = compute_confidence(results)

    user_message = f"""AI SYSTEM DESCRIPTION:
{system_description}

RELEVANT REGULATION EXCERPTS:
{context}

Please provide a compliance assessment."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": user_message}],
        system=SYSTEM_PROMPTS[regulation],
    )

    return response.content[0].text, confidence_score, confidence_label


def compare_systems(description_a, description_b, regulation="EU AI Act"):
    """Compare two AI systems side by side."""
    result_a, conf_a, label_a = classify_system(description_a, regulation)
    result_b, conf_b, label_b = classify_system(description_b, regulation)
    return (result_a, conf_a, label_a), (result_b, conf_b, label_b)