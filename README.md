# AI Compliance Checker — EU AI Act & GDPR

A RAG-powered compliance assessment tool that classifies AI systems against the **EU AI Act (Regulation 2024/1689)** and **GDPR (Regulation 2016/679)** using retrieval-augmented generation and Claude 3.5 Sonnet.

Built as a portfolio project demonstrating practical LLM application development, legal RAG architecture, and EU AI governance tooling.

---

## What it does

Paste a description of any AI system — or upload a model card PDF — and the tool returns:

- **Risk tier classification** (Unacceptable / High / Limited / Minimal for EU AI Act; High / Medium / Low for GDPR)
- **Cited articles** from the actual regulation text with one-line summaries
- **Concrete compliance checklist** of obligations the developer must fulfil
- **Retrieval confidence score** indicating how closely matched the retrieved articles are to the query
- **Important caveats** flagging edge cases and jurisdictional uncertainties

Supports four modes:

- **EU AI Act only** — full four-tier risk classification
- **GDPR only** — lawful basis, DPIA requirement, data subject rights obligations
- **Both** — combined dual-regulation assessment with cross-referenced checklist
- **Compare two systems** — side-by-side assessment of two AI systems

---

## Architecture

User input (text or PDF upload) is embedded locally using sentence-transformers (all-MiniLM-L6-v2) at zero API cost. The embedding is searched against a FAISS vector index containing 222 EU AI Act article chunks and 161 GDPR article chunks. The top 5 retrieved chunks plus a confidence score are passed to Claude 3.5 Sonnet via the Anthropic API with a structured prompt. Claude returns a risk tier, compliance checklist, and caveats. The Streamlit UI renders colour-coded badges and a downloadable assessment report.

**Key design decisions:**

- Chunked by article boundary rather than fixed token size to preserve legal semantic units
- Confidence score derived from FAISS L2 distance of the best-matching chunk — surfaces retrieval quality to the user
- max_tokens set to 2048 to allow complete checklist generation for complex systems
- Zero model weights downloaded — all LLM inference via Anthropic API

---

## Project structure

    b01-euaiact/
    ├── data/
    │   ├── eu_ai_act.pdf           # Source: EUR-Lex (OJ:L_202401689)
    │   ├── gdpr.pdf                # Source: EUR-Lex (CELEX:32016R0679)
    │   ├── faiss_index.bin         # EU AI Act vector index
    │   ├── gdpr_faiss_index.bin    # GDPR vector index
    │   ├── chunks.pkl              # EU AI Act article chunks
    │   └── gdpr_chunks.pkl         # GDPR article chunks
    ├── src/
    │   ├── ingest.py               # PDF parsing + EU AI Act FAISS index builder
    │   ├── gdpr_ingest.py          # PDF parsing + GDPR FAISS index builder
    │   ├── retriever.py            # FAISS retrieval utility
    │   └── classifier.py           # Retrieval + confidence scoring + Claude API
    ├── app/
    │   └── streamlit_app.py        # Streamlit UI (all modes)
    ├── .streamlit/
    │   └── config.toml             # Suppresses transformer watcher warnings
    ├── .gitignore
    ├── requirements.txt
    └── README.md

---

## Setup

**Prerequisites:** Python 3.11, Anaconda, Anthropic API key

**1. Clone the repo and create the environment:**

    git clone https://github.com/your-username/b01-euaiact.git
    cd b01-euaiact
    conda create -n b01-euaiact python=3.11 -y
    conda activate b01-euaiact
    pip install -r requirements.txt

**2. Add your API key:**

    echo ANTHROPIC_API_KEY=your_key_here > .env

**3. Download the regulation PDFs and save to data/:**

- EU AI Act: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689
- GDPR: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679

**4. Build vector indexes (one-time, runs locally, no API cost):**

    python src/ingest.py
    python src/gdpr_ingest.py

**5. Run the app:**

    streamlit run app/streamlit_app.py

---

## Example results

| System | EU AI Act | GDPR |
|---|---|---|
| Real-time facial recognition at airport (law enforcement) | HIGH / UNACCEPTABLE | HIGH |
| CV screening tool trained on historical hiring data | HIGH | HIGH |
| Medical image diagnosis assistant (radiologist support) | HIGH | HIGH |
| Bank credit scoring — automated loan decisions | HIGH | HIGH |
| E-commerce customer service chatbot | LIMITED | MEDIUM |

---

## Limitations

- **Indicative only** — not a substitute for qualified legal advice. Always consult a specialist for compliance decisions.
- **Retrieval quality** — RAG over 383 article chunks covers the core regulation text. Edge cases requiring cross-referencing of recitals, annexes, and implementing acts may be missed.
- **Confidence scoring** — Medium confidence (25–50%) is typical for legal text retrieval and does not indicate incorrect classification.
- **Jurisdiction** — Assessment assumes EU deployment. Member state-specific implementing legislation is not included.
- **Regulation currency** — Based on the EU AI Act as published August 2024 and GDPR as of 2018.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM | Claude 3.5 Sonnet (Anthropic API) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, local) |
| Vector store | FAISS (Facebook AI Similarity Search) |
| PDF parsing | PyMuPDF (fitz) |
| UI | Streamlit |
| Environment | Python 3.11, conda |

---

## Author

Sushil Kumar — MSc AI and Automation, University West, Sweden, 2026
