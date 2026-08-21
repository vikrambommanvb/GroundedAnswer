# Topic 1: The Grounded Answer (Brite Sparks 2026)

An assistant that answers benefits questions in plain language based on the provided policy manual, cites the exact clause relied upon, and responds *"I don't know, here is who to ask"* when the query is out-of-scope.

## Getting Started

### Prerequisites
- Python 3.10+
- (Optional but recommended) Virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

### Installation
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables (if any, e.g., API keys):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Ingestion
Before running queries, ingest the policy manual:
```bash
python ingest.py <path_to_policy_manual>
```

### Running the Assistant
To query the assistant:
```bash
python query.py "your question here"
```

## Running Tests
To run the automated tests:
```bash
pytest
```
