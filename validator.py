import os
import re

# Fallback contact info as defined in guidelines
FALLBACK_CONTACT = "a supervisor at the Department of Household Services or your local district office."

def determine_refusal_contact(clauses):
    """
    Scans the retrieved clauses to determine the most relevant contact or authority.
    If none is found or if the list is empty, returns the fallback contact.
    """
    if not clauses:
        return FALLBACK_CONTACT

    # Check if we have specific parts of the manual that mention specific authorities
    # e.g., reviews go to supervisor / officer, appeals to the Appeals Panel
    has_review_or_appeal = False
    has_supervisor = False
    has_misrep = False

    for c in clauses:
        text = c.get("content", "").lower() + " " + c.get("part_title", "").lower()
        if "appeal" in text or "panel" in text:
            has_review_or_appeal = True
        if "supervisor" in text:
            has_supervisor = True
        if "misrepresentation" in text or "sanction" in text:
            has_misrep = True

    if has_review_or_appeal:
        return "the Calder County Assistance Appeals Panel or a reviewing officer at the Department."
    elif has_supervisor or has_misrep:
        return "a supervisor at the Department of Household Services."
    
    return FALLBACK_CONTACT

def validate_evidence(query, clauses, min_relevance_score=0.15):
    """
    Performs programmatic validation on retrieved evidence to determine if a query is answerable.
    Returns:
        is_answerable (bool)
        reason (str)
        refusal_message (str or None)
    """
    # 1. Clean query
    query_clean = query.strip().lower()

    # 2. Hard check for empty or extremely short/ambiguous queries
    if len(query_clean.split()) < 3:
        contact = FALLBACK_CONTACT
        return False, "Query is too short or ambiguous.", f"I don't know, here is who to ask: {contact}"

    # 3. Check if no clauses were retrieved
    if not clauses:
        contact = FALLBACK_CONTACT
        return False, "No relevant policy clauses found.", f"I don't know, here is who to ask: {contact}"

    # 4. Check if the top relevance score is too low
    top_score = clauses[0].get("score", 0.0)
    if top_score < min_relevance_score:
        contact = FALLBACK_CONTACT
        return False, f"Relevance score ({top_score}) is below threshold ({min_relevance_score}).", f"I don't know, here is who to ask: {contact}"

    # 5. Check if query contains obvious out-of-scope keyword patterns (garbage, pet license, etc.)
    # that would not overlap with policy manual terms but might get low non-zero score
    unrelated_patterns = [
        r"\bgarbage\b", r"\btrash\b", r"\brecycling\b",
        r"\bpet\b", r"\bdog\b", r"\bcat\b", r"\blicense\b", r"\bpermits?\b",
        r"\bweather\b", r"\bparking\b", r"\bbus\b", r"\btransit\b"
    ]
    for pattern in unrelated_patterns:
        if re.search(pattern, query_clean):
            # Verify if the retrieved clauses actually match these words (to prevent false negatives if manual ever did mention it)
            matched_in_policy = False
            for c in clauses[:2]:
                if re.search(pattern, c.get("content", "").lower()):
                    matched_in_policy = True
                    break
            if not matched_in_policy:
                contact = FALLBACK_CONTACT
                return False, f"Query contains out-of-scope term matching pattern '{pattern}'.", f"I don't know, here is who to ask: {contact}"

    # 6. Check for ambiguous generic queries that lack contextual nouns
    # e.g., "Am I eligible?" or "How do I do this?" without any prior context
    # If the top clause has a score that is not extremely high (e.g. < 0.45) and it's very generic
    generic_query_terms = {"eligible", "eligibility", "apply", "application", "award", "payment", "amount"}
    query_words = set(query_clean.split())
    if query_words.issubset(generic_query_terms.union({"am", "i", "how", "do", "what", "is", "the", "for", "a", "of", "to"})):
        if top_score < 0.45:
            contact = FALLBACK_CONTACT
            return False, "Query is too generic and lacks context.", f"I don't know, here is who to ask: {contact}"

    # Query is programmatically deemed answerable (subject to LLM validation)
    return True, "Evidence is sufficient.", None
