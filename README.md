# QPredict
> **Turn past papers into smarter preparation.**

QPredict is an AI-powered Streamlit app that analyzes previous-year examination question papers. Upload PDFs, and QPredict identifies recurring topics, historical trends, and study priorities — all grounded in evidence from your own papers.

**Live demo:** https://qpredict-snke7a4smrvtccsawgji8m.streamlit.app

---

## Features

| Page | What it does |
|------|-------------|
| 📤 Upload | Upload exam PDFs, extract questions automatically |
| 📚 My Papers | View and delete uploaded papers |
| 📊 Dashboard | Priority-ranked topic clusters with year heatmap and trend charts |
| 🔍 Deep Analysis | TF-IDF + semantic similarity analysis per subject |
| 📖 Syllabus | Map questions to syllabus units |
| 📅 Study Plan | Auto-generated day-by-day study schedule |
| ❓ Quiz | Practice MCQ + short-answer quiz from your papers |
| 🤖 Study Assistant | RAG-powered chat — ask questions about your uploaded papers |

---

## How it works

1. **Upload** previous-year PDFs — supports KTU and most Indian university formats
2. **Questions are extracted** automatically using pattern matching (Part A / Part B / Module sections)
3. **Embeddings** are computed using `sentence-transformers` (falls back to TF-IDF if unavailable)
4. **Clustering** groups semantically similar questions into topic clusters
5. **Priority scoring** ranks topics by frequency, year coverage, recency, and consistency
6. **Dashboard** displays ranked topic cards with charts, year timelines, and score breakdowns

---

## Setup (local)

### 1. Clone the repo
```bash
git clone https://github.com/Athmana/QPredict.git
cd QPredict
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
QPredict/
├── app.py                        Home page + app entry point
├── requirements.txt
│
├── pages/
│   ├── 1_📤_Upload.py
│   ├── 2_📚_My_Papers.py
│   ├── 3_📊_Dashboard.py
│   ├── 4_🔍_Deep_Analysis.py
│   ├── 5_📖_Syllabus.py
│   ├── 6_📅_Study_Plan.py
│   ├── 7_❓_Quiz.py
│   └── 8_🤖_Study_Assistant.py
│
├── src/
│   ├── pdf_parser.py             PDF text extraction (PyMuPDF)
│   ├── question_extractor.py     KTU + general question parsing
│   ├── text_cleaner.py           NLP normalization
│   ├── embeddings.py             Sentence Transformer / TF-IDF embeddings
│   ├── similarity.py             Cosine similarity analysis
│   ├── clustering.py             Agglomerative + DBSCAN clustering
│   ├── scoring.py                Historical Priority Score (4 components)
│   ├── trend_analyzer.py         Year-over-year trend classification
│   ├── ui_helpers.py             Reusable Streamlit components + Plotly charts
│   ├── syllabus_mapper.py        Syllabus unit mapping
│   ├── study_planner.py          Study plan generation
│   ├── quiz_generator.py         MCQ + short-answer quiz generation
│   ├── chunker.py                Text chunking for RAG
│   ├── rag_retriever.py          FAISS / NumPy vector index
│   └── rag_assistant.py          RAG-powered LLM assistant
│
├── database/
│   └── database.py               SQLite schema and query functions
│
├── data/                         Runtime only — not committed
│   ├── uploads/                  Uploaded PDFs
│   ├── processed/                Extracted text files
│   └── qpredict.db               SQLite database (auto-created)
│
└── tests/                        Unit tests
```

---

## Optional: AI features (API key required)

The Quiz and Study Assistant pages can use an LLM to generate better questions and synthesized answers:

- **Groq** (free tier available): https://console.groq.com
- **OpenAI**: https://platform.openai.com

Enter your API key in the sidebar on those pages. Without a key, both pages still work in offline mode using your uploaded papers.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | `sentence-transformers` / scikit-learn TF-IDF |
| Vector search | FAISS / NumPy fallback |
| Clustering | scikit-learn AgglomerativeClustering |
| Charts | Plotly |
| Database | SQLite (built-in, no setup needed) |
| LLM | Groq / OpenAI (optional) |

---

## Disclaimer

QPredict''s Historical Priority Score is based on patterns in **your uploaded papers only**. It is **not** a prediction or guarantee of future exam content. All topics in your syllabus remain important.
