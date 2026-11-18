"""Streamlit interface for the PDF RAG chatbot."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from rag_backend import (
    AnswerResult,
    GeminiRequestError,
    MissingAPIKeyError,
    PDFProcessingError,
    PDFRAGBackend,
    normalize_api_key,
)

PROJECT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
load_dotenv(PROJECT_DIR / ".env")

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 1.5rem 1.7rem; border: 1px solid #293044; border-radius: 18px;
            background: linear-gradient(125deg, #171c28 5%, #1b2034 100%); margin-bottom: 1.5rem; }
    .hero h1 { margin: 0 0 .45rem 0; letter-spacing: -.035em; }
    .hero p { color: #b8bfd1; margin: 0; font-size: 1.02rem; }
    .eyebrow { color: #9da5ff; font-size: .78rem; font-weight: 700;
               letter-spacing: .12em; text-transform: uppercase; margin-bottom: .45rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #252b3a; }
    .stButton > button { border-radius: 10px; min-height: 2.65rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_backend() -> PDFRAGBackend:
    return PDFRAGBackend()


def get_api_key() -> str | None:
    key = normalize_api_key(os.getenv("GOOGLE_API_KEY"))
    if key:
        return key
    try:
        return normalize_api_key(st.secrets["GOOGLE_API_KEY"])
    except (KeyError, StreamlitSecretNotFoundError):
        return None


def render_sources(sources: list[Any]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(f"**{source.filename} · page {source.page}**")
            st.caption(source.excerpt)


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["error"])
        else:
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


st.session_state.setdefault("messages", [])
st.session_state.setdefault("confirm_reset", False)
st.session_state.setdefault("upload_notices", [])

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Your documents, understood</div>
      <h1>📚 PDF RAG Chatbot</h1>
      <p>Ask questions and get intelligent answers from your PDF documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    backend = get_backend()
    documents = backend.list_documents()
except Exception as error:
    LOGGER.exception("The local document database could not be opened.")
    st.error(f"The local document database could not be opened: {error}")
    st.stop()

with st.sidebar:
    st.markdown("## About")
    st.caption("Upload PDFs, then ask questions grounded in their contents.")
    st.markdown("---")
    st.markdown(f"### Processed documents · {len(documents)}")
    if documents:
        for document in documents:
            st.markdown(f"**{document['filename']}**")
            st.caption(f"{document['pages']} pages · {document['chunks']} text chunks")
    else:
        st.caption("No documents processed yet.")

    st.markdown("---")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if not st.session_state.confirm_reset:
        if st.button("Reset document database", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("This permanently removes all locally indexed document chunks.")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm reset", type="primary", use_container_width=True):
            try:
                backend.reset_database()
                st.session_state.messages = []
                st.session_state.confirm_reset = False
                st.success("The document database has been reset.")
                st.rerun()
            except Exception as error:
                st.error(f"The database could not be reset: {error}")
        if cancel_col.button("Cancel", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()

    st.markdown("---")
    if not get_api_key():
        st.warning(
            "Gemini API key is not configured. PDF processing works, but add "
            "`GOOGLE_API_KEY` before asking questions."
        )

st.markdown("### Add documents")
for notice in st.session_state.upload_notices:
    if notice["level"] == "success":
        st.success(notice["message"])
    elif notice["level"] == "info":
        st.info(notice["message"])
    else:
        st.error(notice["message"])
st.session_state.upload_notices = []

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Text-based PDFs are supported. Scanned PDFs require OCR, which is not included.",
)

if st.button(
    "Upload and process",
    type="primary",
    disabled=not uploaded_files,
    use_container_width=False,
):
    notices = []
    progress = st.progress(0, text="Preparing documents...")
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            result = backend.process_pdf(uploaded_file.getvalue(), uploaded_file.name)
            if result.duplicate:
                notices.append(
                    {
                        "level": "info",
                        "message": f"{result.filename} is already in the document database; skipped duplicate.",
                    }
                )
            else:
                notices.append(
                    {
                        "level": "success",
                        "message": f"Processed {result.filename} ({result.chunk_count} text chunks).",
                    }
                )
        except PDFProcessingError as error:
            notices.append({"level": "error", "message": str(error)})
        except Exception:
            filename = Path(uploaded_file.name).name
            LOGGER.exception("Processing failed for %s.", filename)
            notices.append(
                {
                    "level": "error",
                    "message": (
                        f"Processing failed for {filename}. "
                        "Check that the file is a valid PDF and try again."
                    ),
                }
            )
        progress.progress(index / len(uploaded_files), text=f"Processed {index} of {len(uploaded_files)}")
    progress.empty()
    st.session_state.upload_notices = notices
    st.rerun()

st.markdown("---")
st.markdown("### Chat with your PDFs")
if not documents:
    st.info("Upload and process at least one PDF to start a conversation.")

for chat_message in st.session_state.messages:
    render_message(chat_message)

question = st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=not documents,
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your PDFs and preparing an answer..."):
                result: AnswerResult = backend.answer_question(question, api_key=get_api_key())
            st.markdown(result.answer)
            render_sources(result.sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                }
            )
        except MissingAPIKeyError as error:
            st.error(str(error))
            st.session_state.messages.append(
                {"role": "assistant", "content": "", "error": str(error)}
            )
        except GeminiRequestError as error:
            st.error(str(error))
            st.session_state.messages.append(
                {"role": "assistant", "content": "", "error": str(error)}
            )
        except Exception:
            message = "Something went wrong while searching the documents. Please try again."
            LOGGER.exception("Unexpected error while answering a document question.")
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": "", "error": message}
            )
