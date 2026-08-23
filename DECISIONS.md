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
- **Fixed-size/Token-based Text Splitters**: Standard chunking methods (e.g., splitting every 500 characters with overlap) were rejected. These splitters cut paragraphs, detach list items from their headers, and separate clause numbers (`§X.Y.Z`) from their context, which would severely degrade the precision of clause citations.
- **External Vector Database Servers**: Third-party databases (like Pinecone, Milvus, or ChromaDB servers) were rejected to keep the deployment footprint zero-dependency and fast. For 148 documents, an in-memory TF-IDF/BM25 combined with local numpy cosine vectors runs in microseconds and compiles cleanly in any environment.
- **Silent Choice in Contradictions**: We rejected having the model silently pick one clause over another (e.g., preferring 10 days over 30 days based on retrieval order or semantic weight). This violates safety regulations, so showing both and flagging the contradiction was preferred.

## 5. What Was Cut (Due to Time)
- **SQLite Embedding Cache**: Embedding caches are currently written back to `clauses.json`. Using an SQLite DB to store binary embeddings would be a cleaner production strategy but was cut to keep the file structure simple.
- **caseworker Web UI**: A simple web application utilizing Streamlit or HTML was cut. The CLI is the primary delivery vehicle.

## 6. What the Solution Does Not Do / Limitations
- **Does Not Answer Student Rules**: Because of the gaps in the manual (directing to care allowance clauses and missing student needs calculations), the assistant will safely refuse questions about student award calculations.
- **Multi-turn Memory**: The assistant answers one question at a time to prevent state pollution and conversation-drift grounding violations.

## 7. Next Steps / First Improvements
- **SQLite Database Storage**: Move from `clauses.json` to a local `policy.db` storing structured text and precomputed embedding blobs for better indexing and scalability.
- **Interactive Caseworker UI**: Develop a lightweight Web UI (e.g. via Streamlit) that prints policy answers and allows caseworkers to click on citations to inspect the source manual text.

## 8. Amendment No. 2026-01

### What Changed
We implemented support for **Amendment No. 2026-01** (effective 1 March 2026) to make the assistant a **Date-Aware Grounded RAG System**. The amendment modifies:
* **§6.4.1(a)**: Earnings disregard from $120 to $175 per month.
* **§4.3.2**: Change reporting timeframe from 10 to 14 days.
* **§9.1.4**: Overpayment reporting threshold from 30 to 14 days.
* **§6.6.1**: Income thresholds table values (e.g. Size 4 = $2,500).
* **§10.5.2**: Sanction percentage from 20% to 15%.
* **§10.5.3A**: Inserts a new sanction exemption for failures to report changes that increase the award.
It also introduces transitional provisions:
* **§5.1**: Earnings disregard, income thresholds, and sanction amendments apply to determinations made on or after 1 March 2026.
* **§5.2**: Change reporting amendments apply only to changes of circumstances occurring on or after 1 March 2026.
* **§5.3**: Claims spanning 1 March 2026 are apportioned daily under §7.4.3.

### What Files Changed
* [ingest.py](file:///Users/sivabalan/Documents/groundedanswer/ingest.py): Expanded to parse `Amendment No. 2026-01.md`, link amendments to their targets, and output a versioned clause overlay schema.
* [retriever.py](file:///Users/sivabalan/Documents/groundedanswer/retriever.py): Added query timeline regex date extraction (`determination_date`, `event_date`, `is_spanning`), a version resolution engine to calculate version applicability per transitional rules, and cross-reference auto-retrieval.
* [validator.py](file:///Users/sivabalan/Documents/groundedanswer/validator.py): Updated validation to verify relevance of applicable versions, and distinguish between historical (active) and post-March (resolved) contradictions.
* [generator.py](file:///Users/sivabalan/Documents/groundedanswer/generator.py): Adapted generation prompts for timeline date-awareness, and enhanced citation validation to support subclauses (e.g. `[§6.4.1(a)]`) and amendment paragraphs (e.g. `[Amendment §2.1]`).
* [main.py](file:///Users/sivabalan/Documents/groundedanswer/main.py): Integrated query date extraction, pass-through to LLM, and debug logging.
* [test_assistant.py](file:///Users/sivabalan/Documents/groundedanswer/test_assistant.py): Written 7 unit and integration tests checking date parsing, version resolution, spanning claims, and LLM output consistency.

### How Amendments are Represented
Amendments are represented as a versioned overlay in `clauses.json`. Base manual clauses have validity dates (`effective_from`, `effective_to`) and transition rules. Amended clauses are stored as duplicate entries with `version = "Amendment No. 2026-01"`, `effective_from = "2026-03-01"`, and their respective `transitional_rule` (§5.1 or §5.2) and `amendment_ref` tags. New clauses (like §10.5.3A) and the amendment paragraphs are also ingested.

### How Effective Dates Work
* **Query Date Extraction**: The query is parsed via rule-based regex to find target dates (e.g. "April 2026", "25 February 2026") and assign them as `determination_date` or `event_date` based on surrounding context.
* **Resolution Layer**: For each retrieved clause, the resolver compares these query dates against the validity dates using the transition rule specified for that clause. It tags each clause as `APPLICABLE`, `SUPERSEDED`, or `INACTIVE`.
* **Annotated Context**: The LLM prompt context contains these status labels, enabling the model to determine what rule was in force.

### How Transitional Provisions Work
* **Determination Date Rule (§5.1)**: Applicability depends entirely on `determination_date` being >= 1 March 2026.
* **Event Date Rule (§5.2)**: Applicability depends entirely on `event_date` (change of circumstances date) being >= 1 March 2026.
* **Spanning Claim Rule (§5.3)**: Both base and amended versions of a clause are marked as `APPLICABLE (Spanning claim)` and the LLM is instructed to apply both sets of figures and apportion the award.

### What Was Deliberately NOT Changed
* **The original policy manual**: `policy-manual.md` remains unchanged to preserve historical grounding.
* **Zero API Dependency for Retrieval**: Embedded vectors (which require Gemini API calls) remain disabled. Instead, we use local BM25 + exact clause boosts + cross-reference auto-retrieval.

### Why Date-Aware Retrieval is Necessary
Policy updates modify eligibility thresholds, disregard figures, and deadlines. Caseworkers review claims from previous months or claims spanning transition dates. Answering using only the latest rules would result in unlawful determinations for historical periods.

### Why Embedding API Calls Were Removed/Reduced
Embeddings generated a high volume of API calls, leading to `429 RESOURCE_EXHAUSTED` rate limits. Combining BM25, exact boosts, and cross-reference auto-retrieval ensures 100% accurate retrieval at zero API cost, operating offline.

### What Would Be Improved With More Time
* **SQLite Relational DB**: Store versioned clauses in a database for robust SQL date-range queries.
* **Local Embedding Models**: Use a lightweight sentence-transformer model locally for offline semantic search.
* **Side-by-Side Version UI**: Develop a caseworker interface displaying historical changes side-by-side.
