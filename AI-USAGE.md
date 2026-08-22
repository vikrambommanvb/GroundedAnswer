# AI Usage Disclosure

## AI Tools Used
- **Antigravity Coding Assistant / IDE (Google Deepmind)**: Used for repository setup, workspace skill configuration, and python scripts scaffolding.

## Key Uses & Scope
- **Project Structure**: Created `README.md`, `DECISIONS.md`, `AI-USAGE.md` templates and configured `.gitignore`.
- **Workspace Customizations**: Enabled a workspace skill to guide RAG architectural choices.
- **Scaffolding**:
  - Drafted and refined the logic for `ingest.py` to parse markdown sections and paragraphs.
  - Wrote the BM25, cosine-similarity, and hybrid retrieval logic in `retriever.py`.
- **Debugging / Verification**: Ran and validated manual parsing checks on the generated `clauses.json` outputs and verified `retriever.py` output via local zsh terminal execution.
