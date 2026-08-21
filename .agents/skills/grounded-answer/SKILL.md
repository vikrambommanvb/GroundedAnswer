---
name: grounded-answer
description: >-
  Use this skill when building, testing, or updating the grounded RAG assistant (Topic 1: The Grounded Answer) for the Brite Sparks 2026 Hackathon. 
  Activate this skill whenever working on policy manuals, clause citation, grounded answers, handling out-of-scope questions, or setting up the project structure (DECISIONS.md, AI-USAGE.md, README.md).
---

# Grounded Answer (RAG) Skill

This skill guides the implementation of **Topic 1: The Grounded Answer** for the Brite Sparks 2026 Hackathon. 

The goal is to build an assistant that:
1. Answers questions in plain language based *only* on the provided policy manual.
2. Cites the exact clause it relied on.
3. Says *"I don't know, here is who to ask"* when the manual does not cover the question.

---

## 1. Project Setup and Submission Compliance

Always ensure the workspace complies with the hackathon's submission guidelines:
*   **Git Repository**: Initialize a Git repository immediately. Commit your work incrementally as you build. Do not have a single final commit.
*   **README.md**: Include clear setup, dependencies, configuration, and run instructions. Verify they work in a clean environment.
*   **DECISIONS.md**: Document all design choices, trade-offs, rejected approaches, cuts for time, limitations, and future improvements. Start this file early.
*   **AI-USAGE.md**: Document the usage of AI tools (including this assistant) honestly and clearly.

---

## 2. Technical Architecture & Implementation Steps

### Step 2.1: Data Ingestion & Parsing
The policy manual changes quarterly and has structured clauses.
*   Parse the policy manual into logical chunks (clauses/sub-clauses) rather than arbitrary text lengths.
*   Ensure each chunk preserves its hierarchical header (e.g., `Clause 4.2.1 - Eligibility criteria for X`).
*   Extract and store metadata: `clause_id`, `clause_title`, `content`, and `version/quarter`.

### Step 2.2: Vector Storage & Hybrid Retrieval
*   Use a lightweight vector database (e.g., SQLite with a vector extension, ChromaDB, or a simple in-memory numpy/cosine index if keeping it zero-dependency).
*   Implement hybrid search: combine semantic vector search (for concept matching) with keyword/BM25 search (for exact clause numbers or specific terms).
*   Retrieve the top $K$ relevant clauses for each query.

### Step 2.3: Grounding and Response Generation
*   Pass the retrieved clauses to the LLM.
*   **Prompting Constraints**:
    *   Instruct the LLM to answer the question *only* using the provided clauses.
    *   Explicitly forbid hallucination or using external knowledge.
    *   Format the response to include citations at the end of relevant sentences (e.g., `[Clause 3.1]`).
*   **Out-of-Bounds Detection**:
    *   If the retrieved clauses do not contain the answer, or if their relevance score is below a certain threshold, force the model to output: *"I don't know, here is who to ask: [appropriate contact/office]"*.
    *   Do not let the model try to guess or extrapolate from missing info.

---

## 3. Reference File Guidelines

### README.md Template
Ensure the README contains:
```markdown
# Topic 1: Grounded Answer RAG

## Setup Instructions
1. Install dependencies: `...`
2. Configure API keys / environment variables: `...`
3. Ingest the policy manual: `python ingest.py <path_to_manual>`

## Run Instructions
- Run the assistant CLI: `python main.py --query "your query"`
```

### DECISIONS.md Template
```markdown
# Architectural Decisions

## Stack Chosen
- Reason: ...

## Retrieval Strategy
- Reason: ...

## Grounding / Out-of-bounds Handling
- Reason: ...

## What We Cut / Future Improvements
- ...
```

### AI-USAGE.md Template
```markdown
# AI Usage Disclosure

- **Tools Used**: Antigravity IDE
- **Scaffolding / Templates**: ...
- **Debugging & Refactoring**: ...
```
