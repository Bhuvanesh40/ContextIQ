import os
import sys
import json
from pathlib import Path
import streamlit as st

# Resolve paths from this file so launching Streamlit from another working
# directory still finds the project modules and data files.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    # The project modules live under src/ and are imported as top-level packages.
    sys.path.insert(0, str(SRC_DIR))

# Import the RAG operations used by the interface; their implementation stays
# in src/ so the UI only coordinates inputs, outputs, and session state.
from rag_pipeline import (
    load_knowledge_base,
    process_pdf,
    generate_rag_answer,
    generate_cross_document_answer,
    detect_knowledge_gap,
    generate_document_comparison,
    INDEX_PATH,
    METADATA_PATH
)
from ingestion.knowledge_base_manager import (
    ingest_pdf,
    clear_knowledge_base
)

# Configure the browser page before creating any Streamlit elements.
st.set_page_config(
    page_title="ContextIQ - Document Intelligence RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit reruns this file after user interaction, so session state keeps
# conversation and index data alive between those reruns.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def init_default_knowledge_base():
    """Ensure sample knowledge base is populated on first launch."""
    # Create and index the bundled sample PDF when no index is available.
    try:
        # Use a bundled sample so a fresh install has something searchable.
        sample_file = ROOT_DIR / "data" / "uploads" / "sample.pdf"
        if not sample_file.exists():
            from create_sample_pdf import main as create_samples
            create_samples()
        # process_pdf extracts, chunks, embeds, indexes, and persists the sample.
        idx, docs = process_pdf(str(sample_file))
        return idx, docs
    except Exception as e:
        return None, []


if "index" not in st.session_state or "documents" not in st.session_state:
    try:
        # Prefer the saved vector store to avoid rebuilding embeddings at startup.
        if Path(INDEX_PATH).exists() and Path(METADATA_PATH).exists():
            idx, docs = load_knowledge_base()
            st.session_state.index = idx
            st.session_state.documents = docs
        else:
            # Bootstrap a usable store when this is the first application run.
            idx, docs = init_default_knowledge_base()
            st.session_state.index = idx
            st.session_state.documents = docs
    except Exception as e:
        # Recover from an unreadable or incomplete saved store by rebuilding it.
        idx, docs = init_default_knowledge_base()
        st.session_state.index = idx
        st.session_state.documents = docs


def refresh_knowledge_base():
    """Reload FAISS index and metadata from disk."""
    # Refresh the session's in-memory index and metadata from disk.
    try:
        # Load the persisted pair together so vector positions continue to match
        # their corresponding metadata records.
        if Path(INDEX_PATH).exists() and Path(METADATA_PATH).exists():
            idx, docs = load_knowledge_base()
            st.session_state.index = idx
            st.session_state.documents = docs
        else:
            # Recreate the sample store if reset or first-run left no saved pair.
            idx, docs = init_default_knowledge_base()
            st.session_state.index = idx
            st.session_state.documents = docs
    except Exception as e:
        st.error(f"Error loading vector store: {e}")


# The sidebar contains system setup and document-management controls.
with st.sidebar:
    st.title("🧠 ContextIQ")
    st.caption("Production Document-Based RAG System")
    st.divider()

    # The key is read from the environment; the password field avoids displaying
    # it in plain text if the user supplies it interactively.
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY missing from environment", icon="⚠️")
        user_key = st.text_input("Enter Gemini API Key", type="password", placeholder="AIzaSy...")
        if user_key:
            # The Gemini client reads this variable when constructing a request.
            os.environ["GEMINI_API_KEY"] = user_key
            st.rerun()

    st.divider()

    # Allow multiple PDFs to be selected before starting a batch ingestion.
    st.subheader("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard text-based PDF documents."
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded. Click below to index:")
        if st.button("📥 Ingest & Index Documents", type="primary", use_container_width=True):
            # Uploaded content is saved under the project's upload directory,
            # which is also where sample/evaluation documents are expected.
            upload_dir = ROOT_DIR / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                # Persist each upload before calling the ingestion pipeline,
                # which accepts a filesystem path rather than Streamlit's object.
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                status_text.text(f"Processing: {uploaded_file.name}...")
                try:
                    # Append this PDF's chunks to the existing shared index.
                    result = ingest_pdf(
                        file_path,
                        chunk_size=1000,
                        overlap=200
                    )
                    if result["status"] == "already_exists":
                        st.info(result["message"])
                    else:
                        st.success(result["message"])
                except Exception as ex:
                    st.error(f"Error indexing {uploaded_file.name}: {ex}")

                # Report batch progress even if an individual document failed.
                progress_bar.progress((i + 1) / len(uploaded_files))

            # Synchronize the UI with the saved store after processing the batch.
            status_text.text("Ingestion completed successfully!")
            refresh_knowledge_base()
            st.rerun()

    st.divider()

    # Summarize the currently loaded metadata, not files that have yet to be indexed.
    st.subheader("📊 Knowledge Base Stats")
    num_chunks = len(st.session_state.documents) if st.session_state.documents else 0
    unique_docs = len({d["source"] for d in st.session_state.documents}) if st.session_state.documents else 0

    col1, col2 = st.columns(2)
    col1.metric("Indexed PDFs", unique_docs)
    col2.metric("Total Chunks", num_chunks)

    st.caption("Embedding Engine: FastEmbed ONNX (`all-MiniLM-L6-v2`)")
    st.caption("Vector Index: FAISS `IndexFlatIP` (Cosine Sim)")

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        # Remove persisted vectors/metadata, then restore the default sample corpus.
        clear_knowledge_base()
        refresh_knowledge_base()
        st.session_state.chat_history = []
        st.warning("Knowledge base reset. Default knowledge reloaded.")
        st.rerun()


# The main workspace separates Q&A, discovery, and evaluation tasks.
st.title("ContextIQ — Document Intelligence & RAG")
st.markdown(
    "Ask questions grounded strictly in your indexed documents with **source citations and page numbers**."
)

tab_chat, tab_discovery, tab_eval = st.tabs([
    "💬 Document Q&A",
    "🔍 Knowledge Discovery",
    "📊 Evaluation Dashboard"
])

# ------------------------------------------
# TAB 1: Document Q&A (Conversational RAG)
# ------------------------------------------
with tab_chat:
    # Warn when selected uploads have not yet been included in the searchable index.
    if uploaded_files:
        uploaded_names = {f.name.lower() for f in uploaded_files}
        indexed_names = {d["source"].lower() for d in st.session_state.documents} if st.session_state.documents else set()
        unindexed = uploaded_names - indexed_names
        if unindexed:
            st.warning(f"⚠️ You have {len(unindexed)} unindexed file(s) in the sidebar. Please click **'📥 Ingest & Index Documents'** to include them in search!")

    col_retrieval, col_search = st.columns([1, 2])
    with col_retrieval:
        # Retrieval controls limit result count and discard weak semantic matches.
        top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=3)
        min_sim = st.slider("Min Similarity Threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

        # Restrict retrieval to one indexed PDF when the user selects a source.
        sources_available = sorted({d["source"] for d in st.session_state.documents}) if st.session_state.documents else []
        selected_source = st.selectbox(
            "Filter by Document (Optional)",
            options=["All Documents"] + sources_available
        )
        source_filter = None if selected_source == "All Documents" else selected_source

    with col_search:
        st.markdown("##### Ask a Question")
        user_query = st.text_input(
            "Enter your question here:",
            placeholder="e.g. What is Retrieval-Augmented Generation?",
            key="user_query_input"
        )
        submit_btn = st.button("🔎 Search & Answer", type="primary")

    if submit_btn and user_query:
        # Do not attempt retrieval or generation without both indexed content and
        # a configured external language-model API key.
        if not st.session_state.index or not st.session_state.documents:
            st.error("Vector store is empty! Initializing default knowledge base...")
            refresh_knowledge_base()
            st.rerun()
        elif not os.getenv("GEMINI_API_KEY"):
            st.error("Gemini API Key is missing! Enter it in the sidebar or set GEMINI_API_KEY in .env.")
        else:
            with st.spinner("Retrieving relevant passages and synthesizing answer..."):
                try:
                    # The pipeline rewrites follow-ups, retrieves evidence, and
                    # asks Gemini to answer using that evidence.
                    result = generate_rag_answer(
                        index=st.session_state.index,
                        documents=st.session_state.documents,
                        question=user_query,
                        top_k=top_k,
                        min_similarity=min_sim,
                        chat_history=st.session_state.chat_history,
                        source_filter=source_filter
                    )

                    # Save both turns so later questions can resolve references.
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

                    # Present the generated response with inline source citations.
                    st.markdown("### 🤖 Answer")
                    st.markdown(result["answer"])

                except Exception as ex:
                    st.error(f"Generation error: {ex}")

    # Replay prior turns after the latest interaction for conversational context.
    if st.session_state.chat_history:
        st.divider()
        st.markdown("#### 📜 Conversation History")
        for msg in reversed(st.session_state.chat_history):
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=role_icon):
                st.write(msg["content"])


# ------------------------------------------
# TAB 2: Knowledge Discovery
# ------------------------------------------
with tab_discovery:
    # This tab offers collection-wide synthesis, coverage checks, and comparisons.
    st.subheader("Cross-Document Synthesis & Gap Detection")
    st.markdown("Leverage advanced multi-document capabilities across your indexed collection.")

    discovery_mode = st.radio(
        "Choose Discovery Mode:",
        ["Cross-Document Synthesis", "Detect Knowledge Gap", "Document Comparison"],
        horizontal=True
    )

    if discovery_mode == "Cross-Document Synthesis":
        cross_q = st.text_input("Question to synthesize across all PDFs:", placeholder="e.g. Compare the concepts explained across files")
        if st.button("Synthesize Answer", type="primary"):
            if not st.session_state.index:
                st.warning("Please index documents first.")
            else:
                with st.spinner("Synthesizing across documents..."):
                    # Search the complete index, then synthesize the retrieved evidence.
                    res = generate_cross_document_answer(
                        st.session_state.index,
                        st.session_state.documents,
                        cross_q
                    )
                    st.markdown(res["answer"])

    elif discovery_mode == "Detect Knowledge Gap":
        gap_q = st.text_input("Check if knowledge base covers this topic:", placeholder="e.g. Does the document mention fine-tuning?")
        if st.button("Check Coverage"):
            if not st.session_state.index:
                st.warning("Please index documents first.")
            else:
                with st.spinner("Analyzing knowledge base coverage..."):
                    # Report retrieved evidence volume and its originating PDFs.
                    gap = detect_knowledge_gap(
                        st.session_state.index,
                        st.session_state.documents,
                        gap_q
                    )
                    st.info(f"**Status:** {gap['status']}")
                    st.write(gap["explanation"])
                    st.write(f"**Retrieved Chunks:** {gap['retrieved_chunk_count']}")
                    st.write(f"**Relevant Sources:** {', '.join(gap['source_documents'])}")

    elif discovery_mode == "Document Comparison":
        if len(sources_available) < 2:
            st.warning("Please index at least two different PDF documents to compare them.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                doc_a = st.selectbox("Document A", sources_available, key="doc_a")
            with col_b:
                doc_b = st.selectbox("Document B", [s for s in sources_available if s != doc_a], key="doc_b")

            comp_q = st.text_input("Comparison Topic / Question:", placeholder="e.g. What are the key differences?")
            if st.button("Compare Documents", type="primary"):
                with st.spinner("Comparing documents..."):
                    try:
                        # Retrieval is constrained independently to each chosen PDF.
                        comp = generate_document_comparison(
                            st.session_state.index,
                            st.session_state.documents,
                            comp_q,
                            doc_a,
                            doc_b
                        )
                        st.markdown(comp["comparison"])
                    except Exception as ex:
                        st.error(str(ex))


# ------------------------------------------
# TAB 3: Evaluation Dashboard
# ------------------------------------------
with tab_eval:
    # Read or regenerate retrieval metrics produced from the labeled evaluation set.
    st.subheader("Retrieval Performance & RAG Metrics")
    st.markdown("Quantitative assessment of ContextIQ's dense retrieval engine across ground-truth evaluation queries.")

    eval_report_file = ROOT_DIR / "data" / "index" / "evaluation_report.json"

    if st.button("▶️ Run Retrieval Benchmark", type="primary"):
        if not st.session_state.index or not st.session_state.documents:
            st.warning("Knowledge base must be indexed to run evaluation.")
        else:
            with st.spinner("Running retrieval evaluation..."):
                try:
                    # Compare unfiltered top-k retrieval against threshold filtering.
                    from evaluation.run_evaluation import evaluate_retrieval
                    from evaluation.evaluation_dataset import evaluation_dataset
                    baseline, b_det = evaluate_retrieval(st.session_state.index, st.session_state.documents, evaluation_dataset, k=3, min_similarity=0.0)
                    filtered, f_det = evaluate_retrieval(st.session_state.index, st.session_state.documents, evaluation_dataset, k=3, min_similarity=0.20)
                    rep = {"baseline": baseline, "filtered": filtered, "baseline_details": b_det, "filtered_details": f_det}
                    with open(eval_report_file, "w", encoding="utf-8") as f:
                        # Store both aggregate metrics and per-query details for display.
                        json.dump(rep, f, indent=4)
                    st.success("Evaluation completed and report updated!")
                except Exception as ex:
                    st.error(f"Evaluation error: {ex}")

    if eval_report_file.exists():
        try:
            # Load the most recently saved benchmark so the dashboard survives reruns.
            with open(eval_report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            b = report_data.get("baseline", {})
            filt = report_data.get("filtered", {})

            st.markdown("#### Retrieval Metrics Summary (K=3)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Precision@3", f"{filt.get('average_precision', 0):.4f}", delta=f"{filt.get('average_precision', 0) - b.get('average_precision', 0):+.4f}")
            m2.metric("Recall@3", f"{filt.get('average_recall', 0):.4f}")
            m3.metric("MRR@3", f"{filt.get('average_mrr', 0):.4f}")
            m4.metric("Hit Rate@3", f"{filt.get('average_hit_rate', 0):.4f}")
            m5.metric("NDCG@3", f"{filt.get('average_ndcg', 0):.4f}")

            st.divider()
            st.markdown("#### Baseline vs. Threshold-Filtered Comparison")
            import pandas as pd
            comp_df = pd.DataFrame([
                {"Metric": "Precision@3", "Raw Top-K (Baseline)": b.get("average_precision", 0), "With Similarity Filter (>=0.20)": filt.get("average_precision", 0)},
                {"Metric": "Recall@3", "Raw Top-K (Baseline)": b.get("average_recall", 0), "With Similarity Filter (>=0.20)": filt.get("average_recall", 0)},
                {"Metric": "MRR@3", "Raw Top-K (Baseline)": b.get("average_mrr", 0), "With Similarity Filter (>=0.20)": filt.get("average_mrr", 0)},
                {"Metric": "Hit Rate@3", "Raw Top-K (Baseline)": b.get("average_hit_rate", 0), "With Similarity Filter (>=0.20)": filt.get("average_hit_rate", 0)},
                {"Metric": "NDCG@3", "Raw Top-K (Baseline)": b.get("average_ndcg", 0), "With Similarity Filter (>=0.20)": filt.get("average_ndcg", 0)},
            ])
            st.dataframe(comp_df, use_container_width=True)

            with st.expander("🔍 View Per-Query Evaluation Details"):
                st.json(report_data.get("filtered_details", []))

            # RAGAS scores are generated separately and loaded if a report exists.
            st.divider()
            st.markdown("### 🎯 RAGAS Generation & Grounding Evaluation")
            st.caption("Measures factual grounding, hallucination prevention, and semantic correctness via Gemini Judge.")

            ragas_report_file = ROOT_DIR / "data" / "index" / "ragas_report.json"

            if ragas_report_file.exists():
                try:
                    with open(ragas_report_file, "r", encoding="utf-8") as rf:
                        ragas_data = json.load(rf)
                    r_summary = ragas_data.get("summary", {})

                    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                    rc1.metric("Faithfulness", f"{r_summary.get('faithfulness', 0):.4f}", help="Checks if claims in the answer are grounded in context (0 hallucinations = 1.0)")
                    rc2.metric("Answer Relevance", f"{r_summary.get('answer_relevancy', 0):.4f}", help="Checks how pertinent the answer is to the question")
                    rc3.metric("Context Precision", f"{r_summary.get('context_precision', 0):.4f}", help="Checks if relevant chunks are ranked at the top")
                    rc4.metric("Context Recall", f"{r_summary.get('context_recall', 0):.4f}", help="Checks if context covers all ground-truth facts")
                    rc5.metric("Answer Correctness", f"{r_summary.get('answer_correctness', 0):.4f}", help="Semantic and factual agreement with ground truth")

                    with st.expander("🔍 View RAGAS Query-Level Records"):
                        st.json(ragas_data.get("records", []))
                except Exception as rex:
                    st.warning(f"Could not load RAGAS report: {rex}")
            else:
                st.info("No RAGAS report found. Run `python src/evaluation/ragas_evaluator.py` to generate generation metrics.")

        except Exception as e:
            st.warning(f"Could not read evaluation report: {e}")
    else:
        st.info("No evaluation report generated yet. Click 'Run Retrieval Benchmark' above to evaluate.")
