# AI Usage Disclosure

## AI Tools Used
- **Antigravity Coding Assistant / IDE (Google Deepmind)**: Used for repository setup, workspace skill configuration, python scripts scaffolding, unit test implementation, and document creation.

## Key Uses & Scope
- **Project Structure**: Created `.gitignore` and `requirements.txt` to isolate local environments and configure dependencies.
- **Workspace Customizations**: Enabled a workspace skill to guide RAG architectural choices.
- **Scaffolding**:
  - Drafted and refined the logic for `ingest.py` to parse markdown sections and paragraphs.
  - Wrote the BM25, cosine-similarity, and hybrid retrieval logic in `retriever.py`.
  - Scaffolding of programmatic validation and contact routing rules in `validator.py`.
  - Constructed the Gemini grounded API calls, low temperature configuration, and post-generation citation validation in `generator.py`.
  - Built the CLI parser and debug output stream in `main.py`.
- **Testing Assistance**: Scaffaffolded the 14-test suite in `test_assistant.py` to mock the Gemini model and test ingestion, retrieval, and validation locally.
- **Documentation**: Formulated the `README.md` containing all 27 required sections and documented design decisions in `DECISIONS.md`.
- **Debugging / Verification**: Ran and validated manual parsing checks on the generated `clauses.json` outputs and verified `retriever.py` and `validator.py` output via local zsh terminal execution, and verified the pytest execution environment.
