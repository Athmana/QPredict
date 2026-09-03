# QPredict
**"Turn past papers into smarter preparation."**

QPredict is an AI-powered education platform that analyzes previous-year examination question papers. It identifies recurring questions, semantically similar questions, important topics, historical trends, and study priorities.

## Current phase
**Phase 1 — PDF Upload and Text Extraction**

---

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

### 4. Run tests
```bash
python -m pytest tests/ -v
```

---

## Project structure

```
qpredict/
│
├── app.py                    Main Streamlit application
├── requirements.txt          Python dependencies
├── README.md
│
├── data/
│   ├── uploads/              Uploaded PDF files (not committed to git)
│   ├── processed/            Extracted text files
│   └── qpredict.db           SQLite database (auto-created)
│
├── src/
│   ├── pdf_parser.py         PDF text extraction (Phase 1)
│   ├── question_extractor.py Question parsing (Phase 2)
│   ├── text_cleaner.py       NLP cleaning (Phase 3)
│   ├── embeddings.py         Sentence embeddings (Phase 4)
│   ├── similarity.py         Similarity calculation (Phase 3-4)
│   ├── clustering.py         Question clustering (Phase 5)
│   ├── topic_extractor.py    Topic labeling (Phase 5-6)
│   ├── trend_analyzer.py     Historical trend analysis (Phase 6)
│   ├── scoring.py            Priority scoring (Phase 6)
│   ├── syllabus_mapper.py    Syllabus analysis (Phase 8)
│   └── study_planner.py      Study plan generation (Phase 9)
│
├── database/
│   └── database.py           SQLite schema and queries
│
├── models/                   Downloaded ML models (not committed)
│
└── tests/
    ├── test_parser.py        Tests for pdf_parser.py
    ├── test_similarity.py    Tests for similarity (Phase 3)
    └── test_scoring.py       Tests for scoring (Phase 6)
```

---

## Development phases

| Phase | Capability | Status |
|-------|-----------|--------|
| 1 | PDF upload + text extraction | ✅ Complete |
| 2 | Question extraction | ⏳ Next |
| 3 | TF-IDF + cosine similarity baseline | — |
| 4 | Sentence embeddings | — |
| 5 | Question clustering | — |
| 6 | Historical priority scoring | — |
| 7 | Full Streamlit dashboard | — |
| 8 | Syllabus intelligence | — |
| 9 | GenAI: study plans, quiz generation | — |
| 10 | RAG + AI Study Assistant | — |

---

## Disclaimer
QPredict's Historical Priority Score is based on patterns found in uploaded examination papers. It is **not** a guaranteed prediction or probability of future examination questions.
