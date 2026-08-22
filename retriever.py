import os
import re
import json
import math
import numpy as np

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
        return cleaned.split()

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
        # Extract clause patterns like 4.3.2 or §4.3.2
        matches = re.findall(r"(?:§\s*)?(\d+\.\d+\.\d+)", text)
        return [f"§{m}" for m in matches]

    def get_semantic_embeddings(self):
        # Only import google-generativeai if API key is active
        import google.generativeai as genai
        
        # Check if we already have embeddings loaded
        if all("embedding" in c and c["embedding"] for c in self.clauses):
            return

        print("Computing semantic embeddings for policy clauses...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
            
        genai.configure(api_key=api_key)
        
        # Batch embed documents
        contents = [f"{c['clause_id']} {c['clause_title']}\n{c['content']}" for c in self.clauses]
        
        # Support batching into chunks of 100 to avoid limits if document count grows
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(contents), batch_size):
            chunk = contents[i:i + batch_size]
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.extend(result["embedding"])
            
        for idx, emb in enumerate(embeddings):
            self.clauses[idx]["embedding"] = emb
            
        # Optionally write back to clauses.json to cache them
        try:
            with open(self.clauses_path, "w", encoding="utf-8") as f:
                json.dump(self.clauses, f, indent=4, ensure_ascii=False)
            print("Successfully cached embeddings in clauses.json.")
        except Exception as e:
            print(f"Warning: Could not cache embeddings: {e}")

    def retrieve(self, query, top_k=TOP_K, semantic_weight=SEMANTIC_WEIGHT, 
                 keyword_weight=KEYWORD_WEIGHT, min_relevance_score=MIN_RELEVANCE_SCORE):
        if not self.clauses:
            return []

        query_tokens = self.tokenize(query)
        bm25_scores = self.compute_bm25_scores(query_tokens)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 0
        normalized_keyword = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in bm25_scores]

        # Direct Clause Match Boosting
        direct_clause_ids = self.extract_clause_ids(query)
        direct_matches = set()
        
        for idx, clause in enumerate(self.clauses):
            if clause["clause_id"] in direct_clause_ids:
                # Give maximum boost to direct clause matching
                normalized_keyword[idx] = 1.5
                direct_matches.add(clause["clause_id"])

        # Semantic Search path
        api_key = os.environ.get("GEMINI_API_KEY")
        semantic_scores = [0.0] * len(self.clauses)
        semantic_active = False

        if api_key:
            try:
                import google.generativeai as genai
                self.get_semantic_embeddings()
                
                # Embed query
                genai.configure(api_key=api_key)
                q_result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                query_emb = np.array(q_result["embedding"])
                
                # Compute cosine similarities
                for idx, clause in enumerate(self.clauses):
                    if "embedding" in clause and clause["embedding"]:
                        c_emb = np.array(clause["embedding"])
                        # Cosine similarity
                        dot = np.dot(c_emb, query_emb)
                        norm_c = np.linalg.norm(c_emb)
                        norm_q = np.linalg.norm(query_emb)
                        if norm_c > 0 and norm_q > 0:
                            # scale cosine from [-1, 1] to [0, 1]
                            similarity = (dot / (norm_c * norm_q) + 1.0) / 2.0
                            semantic_scores[idx] = float(similarity)
                
                semantic_active = True
            except Exception as e:
                print(f"Warning: Semantic retrieval failed, falling back to keyword-only: {e}")
                semantic_active = False

        # Hybrid score aggregation
        results = []
        for idx, clause in enumerate(self.clauses):
            kw_score = normalized_keyword[idx]
            sem_score = semantic_scores[idx]
            
            if semantic_active:
                # Combine weights
                score = (keyword_weight * kw_score) + (semantic_weight * sem_score)
                method = "hybrid"
            else:
                score = kw_score
                method = "keyword"
                
            # If it was a direct clause match override, flag it
            if clause["clause_id"] in direct_matches:
                method = "direct"
                score = max(score, 1.0) # ensure it stays above min score threshold

            results.append({
                "clause_id": clause["clause_id"],
                "clause_title": clause["clause_title"],
                "content": clause["content"],
                "part_title": clause["part_title"],
                "section_title": clause["section_title"],
                "version": clause["version"],
                "score": round(score, 4),
                "retrieval_method": method
            })

        # Filter by minimum relevance threshold
        filtered_results = [r for r in results if r["score"] >= min_relevance_score]

        # Sort descending
        filtered_results.sort(key=lambda x: x["score"], reverse=True)

        return filtered_results[:top_k]
