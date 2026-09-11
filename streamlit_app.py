"""Browser UI for the legal RAG pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from pathlib import Path
from typing import Any, Dict
from dataclasses import replace

import streamlit as st

from arlc.config import get_config
from retrieval import IngestedCorpusLoader, LegalHybridRAGPipeline


st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="L",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading legal documents and building the search index...")
def load_pipeline() -> LegalHybridRAGPipeline:
    """Load and index the corpus once per Streamlit process."""
    config = get_config()
    # The browser UI prioritizes responsiveness; CLI evaluation keeps the full profile.
    config = replace(
        config,
        use_dense_embeddings=False,
        enable_rerank=False,
        enable_final_rerank=False,
        llm_retry_attempts=1,
        llm_timeout_seconds=10.0,
        mock_llm=not config.ui_use_cloud_llm,
    )
    ingest_dir = Path("ingestion/test_ingest_output")
    if not ingest_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed corpus not found at {ingest_dir}. Run `python main.py --ingest` first."
        )

    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    documents = loader.load_all_documents()
    pipeline = LegalHybridRAGPipeline(config=config)
    pipeline.index_corpus(documents)
    return pipeline


def render_result(result: Dict[str, Any]) -> None:
    """Render answer, citations, and retrieval diagnostics."""
    st.subheader("Answer")
    st.write(result.get("answer", "No answer returned."))

    citations = result.get("citations", [])
    if citations:
        st.subheader("Sources")
        for citation in citations:
            st.markdown(
                f"- Document `{citation.get('doc_id', 'unknown')}` - "
                f"physical page `{citation.get('page', 1)}`"
            )
    else:
        st.info("No source citations were returned.")

    metadata = result.get("metadata", {})
    with st.expander("Retrieval details"):
        st.write({
            "route": metadata.get("route"),
            "mode": metadata.get("mode"),
            "chunks_retrieved": metadata.get("chunks_retrieved"),
            "top_score": metadata.get("top_score"),
            "search_queries": metadata.get("search_queries", []),
        })


def main() -> None:
    st.title("Legal RAG Assistant")
    st.caption("Ask questions about the indexed legal documents and inspect the evidence used.")

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("System status")
        st.success("Pipeline ready")
        st.write(f"Documents: {len(pipeline.documents)}")
        st.write(f"Searchable chunks: {len(pipeline.chunks)}")
        st.write(f"Synthesis mode: {pipeline.active_model}")
        st.write(f"Cloud LLM: {'enabled' if pipeline.cfg.ui_use_cloud_llm else 'disabled'}")
        st.divider()
        st.caption("Metadata questions use the deterministic fast path when possible.")

    sample_queries = [
        "What is the claim number for this appeal?",
        "What was the judgment date?",
        "What did the Court order regarding security for costs?",
    ]
    with st.form("question_form", clear_on_submit=False):
        selected_sample = st.selectbox("Try a sample question", ["Choose one..."] + sample_queries)
        query = st.text_area(
            "Your question",
            height=100,
            placeholder="Ask a question about the legal documents...",
        )
        submitted = st.form_submit_button("Ask the legal assistant", type="primary", use_container_width=True)

    if submitted:
        if not query.strip() and selected_sample != "Choose one...":
            query = selected_sample
        if not query.strip():
            st.warning("Enter a question first.")
            return

        normalized_query = query.strip()
        if st.session_state.get("last_query") == normalized_query:
            result = st.session_state["last_result"]
        else:
            with st.spinner("Searching evidence and preparing a grounded answer..."):
                result = pipeline.answer_question(normalized_query)
            st.session_state["last_query"] = normalized_query
            st.session_state["last_result"] = result
        render_result(result)


if __name__ == "__main__":
    main()
