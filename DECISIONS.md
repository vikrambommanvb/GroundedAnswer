# Architectural decisions

This document logs key choices, rejected alternatives, time-based cuts, limitations, and future improvements.

## 1. Stack and Library Choices
- **Language**: Python 3.14+
- **Vector Search / Index**: In-memory hybrid keyword (TF-IDF/exact) and semantic search (using standard library / numpy).
- **LLM/Embeddings**: Gemini API (`gemini-1.5-flash` for generation, `text-embedding-004` for embedding), loaded securely via environment variable `GEMINI_API_KEY`.

## 2. Retrieval Strategy
- **Parsing/Chunking**: Implemented in `ingest.py`. The parser reads `policy-manual.md` line-by-line and splits it into logical paragraph-level chunks based on clause number patterns (`**X.Y.Z**`). This guarantees that each clause remains structurally intact as a single retrieval unit. Bullet lists (e.g. `(a)`, `(b)`) and Markdown tables are fully preserved within their respective clause chunks. Horizontal rules (`---`) separating parts are filtered out.
- **Retrieval Mechanism**: (Planned) Hybrid search combining direct clause ID lookup, term-matching keywords, and cosine-similarity semantic embeddings.

## 3. Grounding & Citation Strategy
- **Grounding constraint**: ...
- **Out-of-bounds Detection**: ...

## 4. What Was Rejected
- ...

## 5. What Was Cut (Due to Time)
- ...

## 6. What the Solution Does Not Do / Limitations
- ...

## 7. Next Steps / First Improvements
- ...
