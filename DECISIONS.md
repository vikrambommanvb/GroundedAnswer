# Architectural decisions

This document logs key choices, rejected alternatives, time-based cuts, limitations, and future improvements.

## 1. Stack and Library Choices
- **Language**: Python 3.14+
- **Vector Search / Index**: In-memory hybrid keyword (TF-IDF/exact) and semantic search (using standard library / numpy).
- **LLM/Embeddings**: Gemini API (`gemini-1.5-flash` for generation, `text-embedding-004` for embedding), loaded securely via environment variable `GEMINI_API_KEY`.

## 2. Retrieval Strategy
- **Parsing/Chunking**: Implemented in `ingest.py`. The parser reads `policy-manual.md` line-by-line and splits it into logical paragraph-level chunks based on clause number patterns (`**X.Y.Z**`). This guarantees that each clause remains structurally intact as a single retrieval unit. Bullet lists (e.g. `(a)`, `(b)`) and Markdown tables are fully preserved within their respective clause chunks. Horizontal rules (`---`) separating parts are filtered out.
- **Retrieval Mechanism**: Implemented in `retriever.py`. It features:
  1. **Direct Clause ID Lookup**: Scans queries for clause references (e.g. `4.3.2`) using regex, boosting direct matches to the top (score `1.5`, method `'direct'`) to guarantee accurate retrieval for explicit references.
  2. **Keyword Retrieval (BM25)**: A pure Python implementation of the BM25 probabilistic relevance algorithm (defaulting to standard parameters `k1=1.5`, `b=0.75`), removing dependencies on external database servers or search indices.
  3. **Semantic Retrieval**: Uses the Gemini `models/text-embedding-004` API to generate embeddings. To avoid latency, it uses batch requests during indexing and attempts to write-back/cache the embeddings directly in `clauses.json`.
  4. **Hybrid Score Aggregation**: Aggregates normalized keyword and cosine-similarity semantic scores using configurable weights (`KEYWORD_WEIGHT = 0.6`, `SEMANTIC_WEIGHT = 0.4`), gracefully falling back to keyword-only search if `GEMINI_API_KEY` is not present.
  5. **Relevance Thresholding**: Filters candidates below `MIN_RELEVANCE_SCORE = 0.15` to detect out-of-scope queries early.

## 3. Grounding & Citation Strategy
- **Grounding constraint**: The LLM will receive only the user's question and the retrieved policy clauses. It will be instructed via system prompts to answer strictly from the text, never infer or extrapolate, and append clause citations like `[§X.Y.Z]`.
- **Out-of-bounds Detection**: A hybrid, two-layered validation system is implemented:
  1. **Layer 1: Programmatic Validation (`validator.py`)**: Filters out queries that are extremely short, have no retrieved clauses, have scores below `0.15`, or contain obvious out-of-scope keyword patterns (e.g., garbage, pet license) before calling the LLM. Generic queries (like "Am I eligible?") are allowed if there are high-scoring policy clauses, enabling the LLM to present general rules without guessing.
  2. **Layer 2: LLM Validation**: If a query passes Layer 1 but the retrieved clauses do not contain the answer, the LLM is instructed to refuse by outputting the refusal string.
- **Refusal / Contact Selection**:
  - We never invent contact info. If the retrieved clauses mention specific roles (e.g., `supervisor`, `reviewing officer`, `Appeals Panel`), we dynamically build a focused contact instruction (e.g., advising to contact a supervisor for misrepresentation cases).
  - Otherwise, we use the fallback: `"a supervisor at the Department of Household Services or your local district office."` which is clearly documented as a fallback and not policy text.

## 4. What Was Rejected
- ...

## 5. What Was Cut (Due to Time)
- ...

## 6. What the Solution Does Not Do / Limitations
- ...

## 7. Next Steps / First Improvements
- ...
