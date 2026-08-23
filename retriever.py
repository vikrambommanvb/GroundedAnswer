import os
import re
import json
import math
import numpy as np

def stem(word):
    word = word.lower()
    
    # 1. Custom mappings for exact matching of key terms in this domain
    overrides = {
        "eligibility": "elig",
        "eligible": "elig",
        "countable": "count",
        "counted": "count",
        "counting": "count",
        "resources": "resourc",
        "resource": "resourc",
        "assets": "asset",
        "savings": "save",
        "saving": "save",
        "limit": "limit",
        "limits": "limit",
        "limitation": "limit",
        "limitations": "limit",
        "threshold": "threshold",
        "thresholds": "threshold",
        "document": "document",
        "documents": "document",
        "documentation": "document",
        "specified": "specifi",
        "specify": "specifi",
        "applying": "appli",
        "application": "appli",
        "applicant": "appli",
        "applies": "appli",
        "apply": "appli",
        "provided": "provid",
        "provide": "provid",
        "supplying": "suppli",
        "supply": "suppli",
        "excluding": "exclud",
        "exclude": "exclud",
        "excluded": "exclud",
        "exclusion": "exclud",
        "earnings": "earn",
        "earned": "earn",
        "earning": "earn",
        "income": "income",
        "incomes": "income",
        "exceed": "exceed",
        "exceeds": "exceed",
        "exceeded": "exceed",
        "cannot": "cannot",
        "unable": "unabl",
    }
    if word in overrides:
        return overrides[word]
        
    # 2. General fallback rules for suffix stripping
    if len(word) > 4:
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith("ss"):
            pass
        elif word.endswith("s") and not word.endswith("us") and not word.endswith("is") and not word.endswith("as"):
            word = word[:-1]
            
        if word.endswith("eed"):
            if word.endswith("ceed"):
                word = word[:-4] + "ceed" # e.g. exceed -> exceed
        elif word.endswith("ing"):
            word = word[:-3]
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
        elif word.endswith("ed"):
            word = word[:-2]
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
        elif word.endswith("ly"):
            word = word[:-2]
        elif word.endswith("ment"):
            word = word[:-4]
            
    return word

def expand_query_tokens(stemmed_tokens):
    expanded = list(stemmed_tokens)
    token_set = set(stemmed_tokens)
    
    # 1. Asset synonyms
    asset_syns = {"asset", "save", "wealth", "properti", "fund", "capit", "saving"}
    if token_set.intersection(asset_syns):
        if "resourc" not in token_set:
            expanded.append("resourc")
            
    # 2. Resources/Assets synonym backward
    if "resourc" in token_set:
        if "asset" not in token_set:
            expanded.append("asset")
            
    # 3. Excluded/Not countable
    exclude_syns = {"exclud", "exempt", "disregard"}
    if token_set.intersection(exclude_syns):
        if "not" not in token_set:
            expanded.append("not")
        if "count" not in token_set:
            expanded.append("count")
            
    # 4. Supply/Provide
    supply_syns = {"provid", "submit", "send", "give", "present"}
    if token_set.intersection(supply_syns):
        if "suppli" not in token_set:
            expanded.append("suppli")
        if "accompani" not in token_set:
            expanded.append("accompani")
            
    # 5. Required/Specified
    req_syns = {"requir", "need", "must", "mandatori", "necessari"}
    if token_set.intersection(req_syns):
        if "specifi" not in token_set:
            expanded.append("specifi")
            
    # 6. Limit/Threshold
    limit_syns = {"limit", "threshold", "maximum", "max", "cap", "exceed", "abov", "more"}
    if token_set.intersection(limit_syns):
        for w in ["exceed", "limit", "threshold"]:
            if w not in token_set:
                expanded.append(w)
                
    # 7. Document/Evidence
    doc_syns = {"document", "paper", "record", "proof", "doc", "specifi"}
    if token_set.intersection(doc_syns):
        if "evidenc" not in token_set:
            expanded.append("evidenc")
            
    # 8. Income/Earnings
    income_syns = {"income", "earn", "wage", "salari"}
    if token_set.intersection(income_syns):
        for w in ["income", "earn"]:
            if w not in token_set:
                expanded.append(w)
                
    # 9. Cannot/Unable
    unable_syns = {"cannot", "unabl", "fail", "failur"}
    if token_set.intersection(unable_syns):
        for w in ["unabl", "cannot"]:
            if w not in token_set:
                expanded.append(w)

    # 10. Broad Eligibility Concept
    elig_syns = {"elig", "criteria", "condit", "qualifi", "guarante"}
    if token_set.intersection(elig_syns):
        for w in ["resourc", "income", "resid", "age", "exclud", "appli", "threshold", "limit"]:
            if w not in token_set:
                expanded.append(w)

    return expanded

# Configurable settings (loaded from environment or defaults)
TOP_K = int(os.environ.get("TOP_K", 5))
SEMANTIC_WEIGHT = float(os.environ.get("SEMANTIC_WEIGHT", 0.4))
KEYWORD_WEIGHT = float(os.environ.get("KEYWORD_WEIGHT", 0.6))
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", 0.15))

class GroundedAnswerRetriever:
    def __init__(self, clauses_path="clauses.json"):
        self.clauses_path = clauses_path
        self.clauses = []
        self.load_clauses()
        self.build_bm25_index()

    def load_clauses(self):
        if os.path.exists(self.clauses_path):
            with open(self.clauses_path, "r", encoding="utf-8") as f:
                self.clauses = json.load(f)
        else:
            print(f"Warning: {self.clauses_path} not found. Ingestion must run first.")
            self.clauses = []

    def clean_text(self, text):
        text = text.lower()
        # Keep letters, numbers, sections marker, and dots in numbers
        text = re.sub(r"[^\w\s§\.]", "", text)
        return text

    def tokenize(self, text):
        cleaned = self.clean_text(text)
        raw_tokens = cleaned.split()
        return [stem(t) for t in raw_tokens]

    def build_bm25_index(self):
        # BM25 Parameters
        self.k1 = 1.5
        self.b = 0.75
        
        self.doc_tokens = []
        self.doc_lengths = []
        self.vocab = set()
        
        # Word DF (Document Frequency)
        self.df = {}
        
        for c in self.clauses:
            # Combine fields to index
            doc_str = f"{c['clause_id']} {c['clause_title']} {c['content']}"
            tokens = self.tokenize(doc_str)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.vocab.update(tokens)
            
            # Count unique terms in doc for DF
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

        self.num_docs = len(self.clauses)
        self.avg_doc_len = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0
        
        # Calculate IDF for each term in vocab
        self.idf = {}
        for term, df_val in self.df.items():
            # Standard BM25 IDF formula
            self.idf[term] = math.log((self.num_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def compute_bm25_scores(self, query_tokens):
        scores = []
        for i in range(self.num_docs):
            score = 0.0
            doc_len = self.doc_lengths[i]
            tokens = self.doc_tokens[i]
            
            # Calculate term frequencies in this document
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
                
            for q_term in query_tokens:
                if q_term in self.idf:
                    f = tf.get(q_term, 0)
                    idf_val = self.idf[q_term]
                    numerator = f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf_val * (numerator / denominator)
            scores.append(score)
        return scores

    def extract_clause_ids(self, text):
        # Find Amendment §X.Y patterns
        amend_matches = re.findall(r"\bAmendment\s+§?\s*(\d+\.\d+)\b", text, re.IGNORECASE)
        amend_ids = [f"Amendment §{m}" for m in amend_matches]
        
        # Find 3-part IDs: §X.Y.Z or X.Y.Z
        matches_3part = re.findall(r"(?:§\s*)?(\d+\.\d+\.\d+)", text)
        ids_3part = [f"§{m}" for m in matches_3part]
        
        # Find 2-part IDs: §X.Y or X.Y (avoiding 3-part subsets)
        matches_2part = re.findall(r"(?:§\s*)?(\d+\.\d+)(?!\.\d+)", text)
        ids_2part = []
        for m in matches_2part:
            pos = text.find(m)
            if pos != -1:
                context_before = text[max(0, pos-15):pos].lower()
                if "amendment" in context_before:
                    continue
            ids_2part.append(f"§{m}")
            
        return list(set(amend_ids + ids_3part + ids_2part))

    def extract_dates_from_query(self, query):
        q = query.lower()
        is_spanning = False
        if "spanning" in q or "spans" in q or "span" in q:
            is_spanning = True
            
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        from datetime import date
        found_dates = []
        
        # Pattern 1: DD Month YYYY or Month DD, YYYY
        pattern_dd_month_yyyy = re.compile(
            r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b"
        )
        for m in pattern_dd_month_yyyy.finditer(q):
            day = int(m.group(1))
            month_name = m.group(2)
            year = int(m.group(3))
            month = months[month_name]
            found_dates.append((m.start(), date(year, month, day)))
            
        pattern_month_dd_yyyy = re.compile(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s+(\d{4})\b"
        )
        for m in pattern_month_dd_yyyy.finditer(q):
            month_name = m.group(1)
            day = int(m.group(2))
            year = int(m.group(3))
            month = months[month_name]
            found_dates.append((m.start(), date(year, month, day)))
            
        # Pattern 3: Month YYYY
        pattern_month_yyyy = re.compile(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b"
        )
        for m in pattern_month_yyyy.finditer(q):
            start_pos = m.start()
            overlap = False
            for fd in found_dates:
                if abs(fd[0] - start_pos) < 10:
                    overlap = True
                    break
            if not overlap:
                month_name = m.group(1)
                year = int(m.group(2))
                month = months[month_name]
                found_dates.append((start_pos, date(year, month, 1)))

        found_dates.sort(key=lambda x: x[0])
        
        determination_date = None
        event_date = None
        
        for pos, dt in found_dates:
            context_before = q[max(0, pos-40):pos]
            context_after = q[pos:min(len(q), pos+40)]
            context = context_before + " " + context_after
            
            if "determination" in context or "determined" in context or "made" in context or "decision" in context:
                determination_date = dt
            elif "change" in context or "occurred" in context or "occurring" in context or "event" in context or "claim" in context:
                event_date = dt
                
        if len(found_dates) == 1:
            dt = found_dates[0][1]
            if "determination" in q or "determined" in q or "made" in q:
                determination_date = dt
            elif "change" in q or "occurred" in q or "occurring" in q or "event" in q:
                event_date = dt
            else:
                determination_date = dt
                event_date = dt
                
        elif len(found_dates) >= 2 and (determination_date is None or event_date is None):
            dt1, dt2 = found_dates[0][1], found_dates[1][1]
            change_pos = q.find("change")
            det_pos = q.find("determination")
            
            if change_pos != -1 or det_pos != -1:
                dist1_c = abs(found_dates[0][0] - change_pos) if change_pos != -1 else 9999
                dist2_c = abs(found_dates[1][0] - change_pos) if change_pos != -1 else 9999
                dist1_d = abs(found_dates[0][0] - det_pos) if det_pos != -1 else 9999
                dist2_d = abs(found_dates[1][0] - det_pos) if det_pos != -1 else 9999
                
                if dist1_c < dist2_c or dist2_d < dist1_d:
                    event_date = dt1
                    determination_date = dt2
                else:
                    event_date = dt2
                    determination_date = dt1
            else:
                event_date = dt1
                determination_date = dt2

        if "after 1 march 2026" in q or "post-amendment" in q or "after march 2026" in q:
            event_date = date(2026, 3, 2)
            determination_date = date(2026, 3, 2)
            
        return determination_date, event_date, is_spanning

    def resolve_applicability(self, retrieved_clauses, determination_date, event_date, is_spanning):
        from datetime import date
        det_dt = determination_date or date(2026, 8, 23)
        ev_dt = event_date or date(2026, 8, 23)
        cutoff = date(2026, 3, 1)

        resolved = []
        for rc in retrieved_clauses:
            c = rc.copy()
            rule = c.get("transitional_rule")
            version = c.get("version")
            cid = c.get("clause_id")
            
            c["is_applicable"] = True
            c["applicability_status"] = "APPLICABLE"
            c["applicability_reason"] = "Always applicable"

            if version == "Amendment No. 2026-01" and cid.startswith("Amendment §"):
                c["is_applicable"] = True
                c["applicability_status"] = "APPLICABLE"
                c["applicability_reason"] = "Amendment text reference"
                resolved.append(c)
                continue

            if rule == "§5.1":
                if is_spanning:
                    c["is_applicable"] = True
                    c["applicability_status"] = "APPLICABLE (Spanning claim)"
                    c["applicability_reason"] = "Claim spans 1 March 2026, figures from both periods apply and are apportioned under §5.3 / §7.4.3."
                else:
                    if det_dt >= cutoff:
                        if version == "Amendment No. 2026-01":
                            c["is_applicable"] = True
                            c["applicability_status"] = "APPLICABLE"
                            c["applicability_reason"] = f"Determination date ({det_dt}) is on or after 1 March 2026, applying the amended rule (per §5.1)."
                        else:
                            c["is_applicable"] = False
                            c["applicability_status"] = "SUPERSEDED"
                            c["applicability_reason"] = f"Superseded for determinations made on or after 1 March 2026 (per §5.1)."
                    else:
                        if version == "Amendment No. 2026-01":
                            c["is_applicable"] = False
                            c["applicability_status"] = "INACTIVE"
                            c["applicability_reason"] = f"Not yet in force for determinations made before 1 March 2026 (per §5.1)."
                        else:
                            c["is_applicable"] = True
                            c["applicability_status"] = "APPLICABLE"
                            c["applicability_reason"] = f"Determination date ({det_dt}) is before 1 March 2026, applying the base rule (per §5.1)."
            elif rule == "§5.2":
                if ev_dt >= cutoff:
                    if version == "Amendment No. 2026-01":
                        c["is_applicable"] = True
                        c["applicability_status"] = "APPLICABLE"
                        c["applicability_reason"] = f"Change occurred on or after 1 March 2026 ({ev_dt}), applying the amended reporting period (per §5.2)."
                    else:
                        c["is_applicable"] = False
                        c["applicability_status"] = "SUPERSEDED"
                        c["applicability_reason"] = f"Superseded for changes occurring on or after 1 March 2026 (per §5.2)."
                else:
                    if version == "Amendment No. 2026-01":
                        c["is_applicable"] = False
                        c["applicability_status"] = "INACTIVE"
                        c["applicability_reason"] = f"Not applicable for changes occurring before 1 March 2026 (per §5.2)."
                    else:
                        c["is_applicable"] = True
                        c["applicability_status"] = "APPLICABLE"
                        c["applicability_reason"] = f"Change occurred before 1 March 2026 ({ev_dt}), applying the base reporting period (per §5.2)."

            resolved.append(c)
        return resolved

    def retrieve(self, query, top_k=TOP_K, semantic_weight=SEMANTIC_WEIGHT, 
                  keyword_weight=KEYWORD_WEIGHT, min_relevance_score=MIN_RELEVANCE_SCORE):
        if not self.clauses:
            return []

        # Parse query dates
        determination_date, event_date, is_spanning = self.extract_dates_from_query(query)

        query_tokens = self.tokenize(query)
        expanded_tokens = expand_query_tokens(query_tokens)
        bm25_scores = self.compute_bm25_scores(expanded_tokens)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 0
        normalized_keyword = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in bm25_scores]

        # Direct Clause Match Boosting
        direct_clause_ids = self.extract_clause_ids(query)
        direct_matches = set()
        
        for idx, clause in enumerate(self.clauses):
            if clause["clause_id"] in direct_clause_ids:
                normalized_keyword[idx] = 1.5
                direct_matches.add(clause["clause_id"])

        results = []
        for idx, clause in enumerate(self.clauses):
            score = normalized_keyword[idx]
            method = "keyword"
                
            if clause["clause_id"] in direct_matches:
                method = "direct"
                score = max(score, 1.0)

            results.append({
                "clause_id": clause["clause_id"],
                "clause_title": clause["clause_title"],
                "content": clause["content"],
                "part_title": clause["part_title"],
                "section_title": clause["section_title"],
                "version": clause["version"],
                "effective_from": clause.get("effective_from"),
                "effective_to": clause.get("effective_to"),
                "transitional_rule": clause.get("transitional_rule"),
                "amendment_ref": clause.get("amendment_ref"),
                "score": round(score, 4),
                "retrieval_method": method
            })

        # Apply programmatic resolution of applicability based on dates
        resolved_results = self.resolve_applicability(results, determination_date, event_date, is_spanning)

        # Filter by minimum relevance threshold
        filtered_results = [r for r in resolved_results if r["score"] >= min_relevance_score]

        # Sort descending by score, prioritizing APPLICABLE ones
        filtered_results.sort(key=lambda x: (x["applicability_status"] != "APPLICABLE", -x["score"]))

        top_results = filtered_results[:top_k]

        # --- Cross-Reference Expansion ---
        cross_ref_ids = set()
        for tr in top_results:
            content_refs = self.extract_clause_ids(tr["content"])
            for ref in content_refs:
                ref_clean = re.sub(r"\s+", "", ref)
                match = re.match(r"^(§\d+\.\d+\.\d+)\([a-z]\)$", ref_clean)
                if match:
                    ref_clean = match.group(1)
                elif ref_clean.lower().startswith("amendment"):
                    ref_clean = re.sub(r"\s+", " ", ref_clean)
                cross_ref_ids.add(ref_clean)

        # Add referenced clauses to results if not already retrieved
        retrieved_ids = {tr["clause_id"] for tr in top_results}
        for ref_id in cross_ref_ids:
            if ref_id not in retrieved_ids:
                ref_clauses = [c for c in self.clauses if c["clause_id"] == ref_id]
                if ref_clauses:
                    ref_resolved = self.resolve_applicability(ref_clauses, determination_date, event_date, is_spanning)
                    for rc in ref_resolved:
                        rc_copy = rc.copy()
                        top_results.append({
                            "clause_id": rc_copy["clause_id"],
                            "clause_title": rc_copy["clause_title"],
                            "content": rc_copy["content"],
                            "part_title": rc_copy["part_title"],
                            "section_title": rc_copy["section_title"],
                            "version": rc_copy["version"],
                            "effective_from": rc_copy.get("effective_from"),
                            "effective_to": rc_copy.get("effective_to"),
                            "transitional_rule": rc_copy.get("transitional_rule"),
                            "amendment_ref": rc_copy.get("amendment_ref"),
                            "applicability_status": rc_copy.get("applicability_status"),
                            "applicability_reason": rc_copy.get("applicability_reason"),
                            "score": 0.5,
                            "retrieval_method": "cross-reference"
                        })
                        retrieved_ids.add(rc_copy["clause_id"])

        return top_results
