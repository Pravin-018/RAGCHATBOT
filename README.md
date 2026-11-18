# PDF RAG Chatbot

A Streamlit application that indexes text from PDF files in a local persistent
ChromaDB collection, retrieves relevant passages with Sentence Transformers,
and asks Google Gemini to answer using those passages only.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Internet access the first time the Sentence Transformers model is loaded

## Run locally

From the project directory, create and activate a virtual environment, install
the dependencies, and start Streamlit:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Set `GOOGLE_API_KEY` in the root `.env` file:

```text
GOOGLE_API_KEY=your_gemini_api_key
```

The `.env` file is ignored by Git. Never commit a real key. PDF indexing works
without the key; Gemini answers require it.

## How it works

1. `app.py` provides multiple-PDF upload, document status, chat history, and
   source excerpts.
2. `rag_backend.py` extracts text page by page with PyPDF, splits it into
   overlapping chunks with LangChain, and embeds chunks using
   `all-MiniLM-L6-v2`.
3. Chunks and page/file metadata are stored in `chroma_db/`. Re-uploading the
   same PDF bytes is detected and skipped.
4. Questions are embedded and matched against the stored chunks. Gemini receives
   only relevant excerpts and is instructed to use the exact not-found response
   when those excerpts do not answer the question.

Scanned/image-only PDFs are reported with a clear message; OCR is not included.
The first indexing run downloads the embedding model and may take a little time.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository. Do not commit `.env`, `chroma_db/`,
   or Streamlit secrets.
2. In Streamlit Community Cloud, create an app from the repository and select
   `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   ```

4. Deploy. Community Cloud installs dependencies from `requirements.txt`.

The app writes its ChromaDB files to the local `chroma_db/` directory. This is
suitable for local use and for a running Cloud instance, but Community Cloud
does not guarantee persistent local disk across rebuilds or restarts. Re-upload
documents after a restart; for durable production storage, use a persistent
volume or managed vector database. Keep API keys in Streamlit Secrets, not in
the repository.
