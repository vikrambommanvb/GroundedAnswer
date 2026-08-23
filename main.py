import sys
import os
import argparse
from retriever import GroundedAnswerRetriever
from validator import validate_evidence, determine_refusal_contact, FALLBACK_CONTACT
from generator import generate_grounded_answer, MissingAPIKeyError

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounded Answer CLI - Assistant for Calder County Household Support Program."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The query/question to ask the assistant."
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Display debugging details (retrieved clauses, scores, and methods)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    query = args.query.strip()
    
    clauses_json_path = "clauses.json"
    if not os.path.exists(clauses_json_path):
        print(f"Error: {clauses_json_path} not found. Please run policy ingestion first:")
        print("  python ingest.py policy-manual.md")
        sys.exit(1)

    if args.debug:
        print(f"[DEBUG] User query: '{query}'")

    # 1. Initialize retriever
    try:
        retriever = GroundedAnswerRetriever(clauses_path=clauses_json_path)
    except Exception as e:
        print(f"Error initializing retrieval index: {e}")
        sys.exit(1)

    # 2. Retrieve relevant clauses
    try:
        det_dt, ev_dt, is_spanning = retriever.extract_dates_from_query(query)
        query_dates = {
            "determination_date": det_dt,
            "event_date": ev_dt,
            "is_spanning": is_spanning
        }
        retrieved_clauses = retriever.retrieve(query)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        sys.exit(1)

    if args.debug:
        print(f"[DEBUG] Timeline Context - Determination: {det_dt}, Event/Change: {ev_dt}, Spanning: {is_spanning}")
        print(f"[DEBUG] Retrieved {len(retrieved_clauses)} candidate clauses:")
        for idx, rc in enumerate(retrieved_clauses):
            status = rc.get('applicability_status', 'APPLICABLE')
            print(f"  {idx+1}. {rc['clause_id']} (Score: {rc['score']}, Method: {rc['retrieval_method']}, Status: {status})")

    # 3. Validate evidence programmatically
    is_answerable, reason, refusal_msg = validate_evidence(query, retrieved_clauses)
    
    if args.debug:
        print(f"[DEBUG] Programmatic validation: answerable={is_answerable}, reason='{reason}'")

    # 4. If programmatic check fails, refuse immediately and skip LLM
    if not is_answerable:
        if args.debug:
            print("[DEBUG] Programmatic filter triggered. Bypassing LLM generation.")
        print(refusal_msg)
        sys.exit(0)

    # 5. Determine dynamic refusal contact from the retrieved clauses
    contact = determine_refusal_contact(retrieved_clauses)
    
    # 6. Generate grounded answer via LLM
    try:
        answer = generate_grounded_answer(
            query=query,
            retrieved_clauses=retrieved_clauses,
            all_clauses=retriever.clauses,
            refusal_contact=contact,
            query_dates=query_dates
        )
        # Print extracted timeline context
        det_dt = query_dates.get("determination_date")
        ev_dt = query_dates.get("event_date")
        is_span = query_dates.get("is_spanning")
        
        if det_dt or ev_dt or is_span:
            timeline_str = f"[Timeline Context: Determination Date = {det_dt or 'Current'}, Event/Change Date = {ev_dt or 'Current'}{', Spanning = Yes' if is_span else ''}]"
            print(timeline_str)
            print("-" * len(timeline_str))
            
        print(answer)
    except MissingAPIKeyError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except Exception as e:
        # Fallback for unexpected generation issues
        print(f"I don't know, here is who to ask: {contact}")
        if args.debug:
            print(f"[DEBUG] Generation exception occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
