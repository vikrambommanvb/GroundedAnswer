# Topic 1: Grounded Answer RAG

## Problem Statement
A county benefits office fields hundreds of repetitive policy questions every week. Staff apply rules from a policy manual that is amended quarterly. Front-line staff frequently cope with manual ambiguities and updates by asking senior colleagues. The office needs a reliable, automated assistant to answer these questions while avoiding incorrect or fabricated policy advice, which can lead to administrative harm and incorrect determinations.

## Goal
Build a command-line **Date-Aware Grounded RAG** assistant for the Calder County Household Support Program that:
1. Answers questions in plain language based *only* on the provided policy manual and any amendment overlays (such as Amendment No. 2026-01).
2. Dynamically resolves which policy version applies based on the date of the claim/event/determination mentioned in the query.
3. Cites the exact clause and/or amendment paragraph relied upon (e.g. `[§4.3.2]` or `[Amendment §2.1]`).
4. Refuses to answer when the manual does not cover the question or covers it ambiguously, indicating the correct authority to contact.
5. Identifies internal policy contradictions historically and dynamically (reporting pre-amendment contradictions while recognizing when amendments have resolved them).

## Features
- **Date-Aware Version Overlay Resolution**: Automatically parses dates from queries to determine `determination_date` and `event_date`, applying transitional rules (§5.1, §5.2, §5.3) to resolve which clause versions are active.
- **Cross-Reference Auto-Retrieval**: Automatically expands the context when retrieved amendment paragraphs refer to other base clauses (e.g., pulling in §6.4.1 when Amendment §1.1 is retrieved), guaranteeing complete grounding and validation.
- **Logical Clause Parsing**: Instead of arbitrary text splitting, the ingestion parses the manual by logical paragraphs and parts, keeping tabular data and list structures intact.
- **Hybrid Keyword Search**: A native Python BM25 index that works completely offline without API keys.
- **Direct Clause ID Boost**: Scans queries for direct clause references (e.g. `4.3.2` or `§4.3.2` or `Amendment §2.1`) and overrides standard scores to guarantee direct delivery.
- **Evidence Validation Guardrails**: A hard programmatic validation layer (`validator.py`) filters out generic queries, low-relevance results, and obvious out-of-scope keywords before any LLM execution occurs.
- **Contradiction Management**: Instructs the LLM to highlight conflicts, outputting details and citations for both conflicting rules.
- **Citation Guardrail Validation**: Performs regex checks on generated answers to verify that all cited clauses exist in the database and were actually provided as retrieved context, failing back to refusal if validation fails.
- **Dynamic Contact Routing**: Adjusts the refusal contact information dynamically based on query context (e.g. directing to the appeals panel or a supervisor).

## Architecture
```
                     policy-manual.md
                           |
                           v
                       ingest.py
                           |
                           v
                      clauses.json
                           |
                +----------+----------+
                |                     |
                v                     v
       Keyword Retrieval      Semantic Retrieval
            (BM25)            (Gemini Embeddings)
                |                     |
                +----------+----------+
                           |
                           v
                     Hybrid Ranking
                           |
                           v
                         Top-K
                           |
                           v
                   Evidence Check (validator.py)
                           |
                  +--------+--------+
                  |                 |
             Insufficient        Sufficient
                  |                 |
                  v                 v
               REFUSAL        Conflict Check
                                    |
                           +--------+--------+
                           |                 |
                        Conflict         No Conflict
                           |                 |
                           v                 v
                    Explain Conflict   Grounded Prompt
                           |                 |
                           +--------+--------+
                                    |
                                    v
                                 Gemini
                                    |
                                    v
                           Citation Validation
                           (Post-Generation Guard)
                                    |
                           +--------+--------+
                           |                 |
                        Invalid            Valid
                           |                 |
                           v                 v
                         REFUSE             ANSWER
                                             |
                                             v
                                      [§X.X.X] citations
```

## Technology Stack
- **Programming Language**: Python
- **LLM API Provider**: Groq API (via standard HTTP `requests`)
- **Libraries**: `requests` (HTTP client), `numpy` (vector operations), `python-dotenv` (configuration), `pytest` (testing).

## Technology Definitions
- **BM25**: A probabilistic search algorithm that calculates the relevance of documents to a given search query based on term frequencies and document lengths.
- **Retrieval-Augmented Generation (RAG)**: An architectural pattern that retrieves relevant factual fragments from a document corpus and feeds them to an LLM to generate highly grounded, context-aware answers.
- **Grounded Answer**: An LLM-generated response whose statements are strictly derived from and referenced to retrieved context source clauses, minimizing hallucinations.

## Data Source
- **Calder County Household Support Program Policy Manual** (`policy-manual.md`): A consolidated Markdown manual dated 31 December 2025 containing twelve Parts describing general conditions, eligibility, income thresholds, award calculations, overpayments, review, and appeal.

## Input
- A plain-language query string passed as a command-line argument.

## Output
- A grounded, cited, plain-language answer string, or a formatted refusal instruction.

## Project Structure
```
groundedanswer/
├── .agents/                    # Custom agent workspace skills
├── 1/                          # Raw extracted zip data packs & documents
├── clauses.json                # Ingested and structured policy manual
├── ingest.py                   # Parsing script to convert MD to JSON
├── retriever.py                # BM25 keyword and semantic vector search
├── validator.py                # Programmatic validation and out-of-scope checks
├── generator.py                # Gemini generation and citation guardrails
├── main.py                     # Command-line interface with debug logging
├── requirements.txt            # Project python dependencies
├── test_assistant.py           # Pytest unit and integration test suite
├── DECISIONS.md                # Architectural decisions log
├── AI-USAGE.md                 # AI usage disclosure
└── .gitignore                  # Git untracked configuration
```

## Installation
Clone this repository and navigate to the directory:
```bash
git clone https://github.com/vikrambommanvb/GroundedAnswer.git
cd GroundedAnswer
```

## Python Version
- **Minimum Requirement**: Python 3.10+ (Developed and tested with Python 3.14)

## Virtual Environment
To keep dependencies isolated:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Dependencies
Install the required python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## API Key Configuration
Create a `.env` file in the root of the project to configure your Groq API key:
```env
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-20b"
```
*(Note: `.env` is listed in `.gitignore` to prevent credentials from being committed).*

---

## Policy Ingestion
Parse the policy manual markdown and the amendment overlay into version-aware JSON:
```bash
python ingest.py policy-manual.md
```
*(Optionally specify an amendment path: `python ingest.py policy-manual.md "1 - The Grounded Answer/Amendment No. 2026-01.md"`).*
This outputs `clauses.json` containing 163 version-controlled clauses.

---

## Running the Assistant
To run the assistant, use:
```bash
python main.py --query "your question"
```

To run with full debug traces showing extracted query dates and clause applicability statuses:
```bash
python main.py --debug --query "your question"
```

---

## Testing
Run the comprehensive automated test suite containing 21 unit and integration tests covering date-extraction, resolution, spanning period calculations, and citation validation:
```bash
pytest test_assistant.py -v
```
*(Note: Tests that utilize LLM calls mock the Gemini API, meaning the test suite runs successfully without requiring an active API key).*

---

## Example Queries & Expected Answers

### 1. Historical Determination (Pre-March 2026)
* **Query**: `What was the earnings disregard under a determination made in February 2026?`
* **Expected Answer**: $120/month. Under §5.1, the earnings disregard increase ($175) only applies to determinations on or after 1 March 2026.
* **Citations**: `[§6.4.1(a)]` (or parent `[§6.4.1]`) and transitional rule `[§5.1]`.

### 2. Current/Post-Amendment Determination (Post-March 2026)
* **Query**: `What is the earnings disregard for a determination made in April 2026?`
* **Expected Answer**: $175/month. Applies per §5.1 since determination date >= 1 March 2026.
* **Citations**: `[§6.4.1(a)]` (or parent `[§6.4.1]`), `[Amendment §1.1]`, and transitional rule `[§5.1]`.

### 3. Historical Change Event (Pre-March 2026)
* **Query**: `What was the reporting deadline for a change occurring on 20 February 2026?`
* **Expected Answer**: 10 calendar days. Under §5.2, reporting changes amendments apply ONLY to changes occurring on or after 1 March 2026, regardless of the determination date.
* **Citations**: `[§4.3.2]` and `[§5.2]`.

### 4. Post-Amendment Change Event (Post-March 2026)
* **Query**: `What is the reporting deadline for a change occurring on 10 April 2026?`
* **Expected Answer**: 14 calendar days. Applies per §5.2 since event date >= 1 March 2026.
* **Citations**: `[§4.3.2]`, `[Amendment §2.1]`, and `[§5.2]`.

### 5. Mixed Dates (Rule §5.2 Override)
* **Query**: `A change occurred on 25 February 2026 but the determination was made on 10 March 2026. Which reporting period applies?`
* **Expected Answer**: 10 calendar days. Because the change event date (25 Feb 2026) is pre-March, the old rule applies per transitional provision §5.2.
* **Citations**: `[§4.3.2]` and `[§5.2]`.

### 6. Spanning Periods (Rule §5.3 Apportionment)
* **Query**: `What happens to a claim spanning 1 March 2026?`
* **Expected Answer**: Use the figures in force on each day of the spanning period and apportion the award daily under §7.4.3.
* **Citations**: `[§5.3]` and `[§7.4.3]`.

### 7. Historical vs. Resolved Contradictions
* **Historical query (e.g. change in Feb 2026)**: The assistant highlights the contradiction between base §4.3.2 (10 days) and base §9.1.4 (30 days) and outputs refusal contacts.
* **Post-March query (e.g. change in April 2026)**: No contradiction is reported because the amendment aligned both to 14 days.

---

## Safety & Grounding Guardrails
* **No Gemini Embedding API dependency**: Retrieval works completely offline without API keys, protecting against rate limit blockages.
* **Programmatic Validation (`validator.py`)**: Filters ambiguous/short inputs and out-of-scope keywords before invoking LLM.
* **Citation Guardrail Validation**: Confirms that cited paragraphs exist in the manual and were actually provided in the context, failing back to refusal if validation fails.
* **Dynamic Refusal Contact Routing**: Routes refusal responses to specific authorities (appeals panel, supervisor) based on query context.
