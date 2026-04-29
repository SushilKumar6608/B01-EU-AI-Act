#streamlit_app.py
import sys
import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from classifier import classify_system, compare_systems, extract_text_from_pdf

st.set_page_config(
    page_title="EU AI Act & GDPR Compliance Checker",
    page_icon="⚖️",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    regulation = st.selectbox(
        "Regulation",
        ["EU AI Act", "GDPR", "Both"],
        help="Choose which regulation to assess against.",
    )

    mode = st.radio(
        "Mode",
        ["Single system", "Compare two systems"],
        help="Analyse one system or compare two side by side.",
    )

    st.divider()
    st.caption("Built on Regulation (EU) 2024/1689 and Regulation (EU) 2016/679.")
    st.caption("Indicative assessments only — not legal advice.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚖️ AI Compliance Checker")
st.markdown(
    "Assess AI systems against the **EU AI Act**, **GDPR**, or **both**. "
    "Paste a description, upload a model card PDF, or compare two systems."
)
st.divider()

# ── Example descriptions ──────────────────────────────────────────────────────
EXAMPLES = {
    "Select an example...": "",
    "Airport facial recognition (law enforcement)": (
        "A computer vision system deployed at airport security checkpoints that uses "
        "real-time facial recognition to match travellers against a watchlist of "
        "individuals flagged by law enforcement. The system operates continuously "
        "and flags potential matches for human security officers to review."
    ),
    "CV screening tool (recruitment)": (
        "An AI system used by HR departments to automatically screen and rank job "
        "applicants based on their CVs. The system analyses education, work experience, "
        "and skills to produce a shortlist of candidates for human recruiters to review. "
        "It is trained on historical hiring data from the company."
    ),
    "Medical image diagnosis assistant": (
        "A deep learning model that analyses chest X-rays to detect potential signs "
        "of pneumonia and other lung conditions. The system is used in hospitals to "
        "assist radiologists by flagging suspicious areas on scans. Final diagnosis "
        "is always made by a licensed physician."
    ),
    "Customer service chatbot": (
        "A conversational AI assistant deployed on an e-commerce website to answer "
        "customer questions about orders, returns, and products. The chatbot can "
        "escalate complex issues to human agents. It does not make any decisions "
        "about customers and has no access to financial or personal data beyond "
        "order history."
    ),
    "Credit scoring system": (
        "An AI model used by a bank to assess the creditworthiness of loan applicants. "
        "The system analyses financial history, income, and behavioural data to produce "
        "a credit score that directly determines whether an applicant is approved or "
        "rejected for a personal loan."
    ),
}


def render_confidence(score, label):
    if score >= 70:
        st.success(f"🎯 Retrieval Confidence: **{label}** ({score}%)")
    elif score >= 40:
        st.warning(f"⚠️ Retrieval Confidence: **{label}** ({score}%)")
    else:
        st.error(f"❌ Retrieval Confidence: **{label}** ({score}%)")


def render_risk_badge(result_text, regulation):
    """Extract and display a colour-coded risk badge."""
    result_upper = result_text.upper()
    if regulation in ["EU AI Act", "Both"]:
        if "UNACCEPTABLE" in result_upper:
            st.error("🚫 EU AI Act: UNACCEPTABLE RISK — Prohibited under Article 5")
        elif "HIGH" in result_upper:
            st.warning("⚠️ EU AI Act: HIGH RISK — Strict obligations apply")
        elif "LIMITED" in result_upper:
            st.info("ℹ️ EU AI Act: LIMITED RISK — Transparency obligations apply")
        elif "MINIMAL" in result_upper:
            st.success("✅ EU AI Act: MINIMAL RISK — No specific obligations")
    if regulation in ["GDPR", "Both"]:
        if "GDPR RISK LEVEL: HIGH" in result_upper:
            st.error("🔴 GDPR: HIGH risk — DPIA likely required")
        elif "GDPR RISK LEVEL: MEDIUM" in result_upper:
            st.warning("🟡 GDPR: MEDIUM risk — Review data processing obligations")
        elif "GDPR RISK LEVEL: LOW" in result_upper:
            st.success("🟢 GDPR: LOW risk — Standard obligations apply")


# ── SINGLE MODE ───────────────────────────────────────────────────────────────
if mode == "Single system":

    col_ex, col_upload = st.columns([2, 1])

    with col_ex:
        selected = st.selectbox("Load an example:", list(EXAMPLES.keys()))

    with col_upload:
        uploaded_file = st.file_uploader(
            "Or upload a model card PDF",
            type=["pdf"],
            help="Upload a PDF model card and its text will be extracted automatically.",
        )

    # Determine input text
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file.read())
        user_input = st.text_area(
            "Extracted text (edit if needed):",
            value=pdf_text[:3000],  # cap to avoid overwhelming the model
            height=200,
        )
        st.caption(f"Extracted {len(pdf_text):,} characters. Showing first 3,000 for analysis.")
    else:
        user_input = st.text_area(
            "AI System Description",
            value=EXAMPLES[selected],
            height=180,
            placeholder=(
                "Describe your AI system: what it does, where it is deployed, "
                "who operates it, and what decisions it influences..."
            ),
        )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyse = st.button("Analyse", type="primary", use_container_width=True)
    with col2:
        st.caption(f"One API call to Claude Sonnet · Regulation: {regulation}")

    st.divider()

    if analyse:
        if not user_input.strip():
            st.warning("Please enter or upload a system description.")
        else:
            with st.spinner("Retrieving relevant articles and analysing..."):
                try:
                    result, conf_score, conf_label = classify_system(
                        user_input, regulation
                    )
                    render_confidence(conf_score, conf_label)
                    render_risk_badge(result, regulation)
                    st.markdown("### Full Assessment")
                    st.markdown(result)
                    st.divider()
                    st.download_button(
                        "Download Assessment (.txt)",
                        data=result,
                        file_name="compliance_assessment.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ── COMPARE MODE ──────────────────────────────────────────────────────────────
else:
    st.subheader("Compare Two AI Systems")
    st.caption(
        "Describe two systems below. Each will be assessed independently "
        "and displayed side by side so you can compare their risk profiles."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### System A")
        upload_a = st.file_uploader("Upload PDF (System A)", type=["pdf"], key="pdf_a")
        if upload_a:
            text_a = extract_text_from_pdf(upload_a.read())
            desc_a = st.text_area("System A (extracted)", value=text_a[:3000], height=180, key="desc_a")
        else:
            desc_a = st.text_area(
                "System A description",
                height=180,
                placeholder="Describe System A...",
                key="desc_a",
            )

    with col_b:
        st.markdown("#### System B")
        upload_b = st.file_uploader("Upload PDF (System B)", type=["pdf"], key="pdf_b")
        if upload_b:
            text_b = extract_text_from_pdf(upload_b.read())
            desc_b = st.text_area("System B (extracted)", value=text_b[:3000], height=180, key="desc_b")
        else:
            desc_b = st.text_area(
                "System B description",
                height=180,
                placeholder="Describe System B...",
                key="desc_b",
            )

    col1, col2 = st.columns([1, 5])
    with col1:
        compare = st.button("Compare", type="primary", use_container_width=True)
    with col2:
        st.caption(f"Two API calls to Claude Sonnet · Regulation: {regulation}")

    st.divider()

    if compare:
        if not desc_a.strip() or not desc_b.strip():
            st.warning("Please provide descriptions for both systems.")
        else:
            with st.spinner("Analysing both systems..."):
                try:
                    (res_a, conf_a, label_a), (res_b, conf_b, label_b) = compare_systems(
                        desc_a, desc_b, regulation
                    )

                    col_r_a, col_r_b = st.columns(2)

                    with col_r_a:
                        st.markdown("#### System A Results")
                        render_confidence(conf_a, label_a)
                        render_risk_badge(res_a, regulation)
                        st.markdown(res_a)
                        st.download_button(
                            "Download System A (.txt)",
                            data=res_a,
                            file_name="system_a_assessment.txt",
                            mime="text/plain",
                            key="dl_a",
                        )

                    with col_r_b:
                        st.markdown("#### System B Results")
                        render_confidence(conf_b, label_b)
                        render_risk_badge(res_b, regulation)
                        st.markdown(res_b)
                        st.download_button(
                            "Download System B (.txt)",
                            data=res_b,
                            file_name="system_b_assessment.txt",
                            mime="text/plain",
                            key="dl_b",
                        )

                except Exception as e:
                    st.error(f"Error: {str(e)}")