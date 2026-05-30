# GraphRAG Exam Test Builder

Streamlit app for uploading study PDFs, building a lightweight knowledge graph, asking graph-guided RAG questions, and generating exam tests with the Mistral AI API.

## Features

- PDF upload and text extraction
- Sentence-aware chunking
- Entity and relation extraction
- Knowledge graph construction with community detection
- Hybrid retrieval using graph paths plus TF-IDF vector similarity
- Mistral AI answer generation
- Exam generation with MCQs, short answers, and answer key

## Run Locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Set your API key before running:

```bash
set MISTRAL_API_KEY=your_api_key_here
```

PowerShell:

```powershell
$env:MISTRAL_API_KEY="your_api_key_here"
streamlit run app.py
```

## Deploy Free on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from the repository.
3. Set the main file path to `app.py`.
4. Add this secret in app settings:

```toml
MISTRAL_API_KEY = "your_api_key_here"
```

5. Deploy.

Do not commit `.streamlit/secrets.toml`. Use Streamlit Cloud app settings for secrets.

## How the GraphRAG Flow Works

During upload:

```text
PDF -> text extraction -> chunks -> spaCy entities/noun phrases -> co-occurrence relations -> knowledge graph -> communities -> embeddings/vectors
```

During question answering:

```text
question -> entity detection -> community retrieval -> graph traversal -> chunk retrieval -> Mistral answer
```

This keeps retrieval focused on concept relationships, not only keyword or vector similarity.

Entity extraction uses a FastGraphRAG-style traditional NLP path: spaCy named entities such as PERSON, ORG, and GPE plus noun phrases are indexed locally. If spaCy or `en_core_web_sm` is unavailable, the app falls back to the lightweight regex extractor so uploads still work.
