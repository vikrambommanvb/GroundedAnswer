import sys
import os
import re
import json

def parse_policy_manual(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clauses = []
    current_clause = None
    part_title = ""
    section_title = ""
    version = ""
    source_order = 0
    unique_sections = set()

    # Regex patterns
    part_pattern = re.compile(r"^#\s+(Part\s+\d+\s+—\s+.+)$")
    section_pattern = re.compile(r"^##\s+(\d+\.\d+\s+.+)$")
    version_pattern = re.compile(r"Consolidated text as at\s+([^*]+)")
    
    # Matches bold paragraph numbers like **1.1.1** or **1.4.1 Applicant**
    clause_pattern = re.compile(r"^\s*\*\*(\d+\.\d+\.\d+)(?:\s+([^*]+))?\*\*\s*(.*)$")

    # First attempt to extract version from top headers
    for line in lines[:20]:
        v_match = version_pattern.search(line)
        if v_match:
            version = v_match.group(1).strip()
            break

    if not version:
        version = "Unknown Version"

    for line_num, line_raw in enumerate(lines, 1):
        line_stripped = line_raw.strip()

        # Check for Part header
        part_match = part_pattern.match(line_stripped)
        if part_match:
            part_title = part_match.group(1).strip()
            continue

        # Check for Section header
        section_match = section_pattern.match(line_stripped)
        if section_match:
            section_title = section_match.group(1).strip()
            unique_sections.add(section_title)
            continue

        # Ignore horizontal rules
        if line_stripped == "---":
            continue

        # Check for Paragraph/Clause start
        clause_match = clause_pattern.match(line_raw)
        if clause_match:
            # Save the previous clause
            if current_clause:
                current_clause["content"] = current_clause["content"].strip()
                clauses.append(current_clause)

            source_order += 1
            clause_num = clause_match.group(1).strip()
            clause_sub_title = clause_match.group(2)
            if clause_sub_title:
                clause_sub_title = clause_sub_title.strip()
            
            clause_title = clause_sub_title if clause_sub_title else section_title
            
            # The rest of the line content
            rest_content = clause_match.group(3).strip()

            current_clause = {
                "clause_id": f"§{clause_num}",
                "clause_title": clause_title,
                "content": rest_content,
                "part_title": part_title,
                "section_title": section_title,
                "version": version,
                "source_order": source_order
            }
            continue

        # If we are already inside a clause and it's not a heading, append the line
        if current_clause:
            # We preserve formatting by appending with newline
            if current_clause["content"]:
                current_clause["content"] += "\n" + line_raw.rstrip()
            else:
                current_clause["content"] += line_raw.strip()

    # Append the last clause
    if current_clause:
        current_clause["content"] = current_clause["content"].strip()
        clauses.append(current_clause)

    return clauses, len(unique_sections), version

def apply_amendment_layer(clauses, amendment_path):
    # Set default version metadata for base clauses
    for c in clauses:
        c["effective_from"] = "2025-12-31"
        c["effective_to"] = None
        c["transitional_rule"] = None
        c["amendment_ref"] = None

    if not os.path.exists(amendment_path):
        print(f"Warning: Amendment file '{amendment_path}' not found. No amendment layer will be applied.")
        return clauses

    print(f"Applying amendment layer from: {amendment_path}")
    with open(amendment_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse amendment clauses
    amendment_clauses = []
    part_title = "Amendment No. 2026-01"
    section_title = ""
    current_clause = None
    source_order = len(clauses)
    
    clause_pattern = re.compile(r"^\s*\*\*(\d+\.\d+)(?:\s+([^*]+))?\*\*\s*(.*)$")
    
    for line_raw in lines:
        line_stripped = line_raw.strip()
        if line_stripped.startswith("## "):
            section_title = line_stripped[3:].strip()
            continue
        
        clause_match = clause_pattern.match(line_raw)
        if clause_match:
            if current_clause:
                current_clause["content"] = current_clause["content"].strip()
                amendment_clauses.append(current_clause)
            
            source_order += 1
            clause_num = clause_match.group(1).strip()
            rest_content = clause_match.group(3).strip()
            
            current_clause = {
                "clause_id": f"Amendment §{clause_num}",
                "clause_title": f"Amendment §{clause_num} - {section_title}",
                "content": rest_content,
                "part_title": part_title,
                "section_title": section_title,
                "version": "Amendment No. 2026-01",
                "source_order": source_order,
                "effective_from": "2026-03-01",
                "effective_to": None,
                "transitional_rule": None,
                "amendment_ref": None
            }
            continue
            
        if current_clause:
            if line_stripped == "---":
                continue
            if current_clause["content"]:
                current_clause["content"] += "\n" + line_raw.rstrip()
            else:
                current_clause["content"] += line_raw.strip()
                
    if current_clause:
        current_clause["content"] = current_clause["content"].strip()
        amendment_clauses.append(current_clause)

    # Convert transitional rules to appropriate clause IDs (§5.1, §5.2, §5.3)
    for ac in amendment_clauses:
        if "5. Transitional provision" in ac["section_title"]:
            num = ac["clause_id"].split("§")[1]
            ac["clause_id"] = f"§{num}"
            ac["clause_title"] = f"Transitional Provision §{num}"
            ac["section_title"] = "5. Transitional provision"

    # Overlay modification logic
    modified_clauses = []
    
    def find_clause(clause_id):
        for c in clauses:
            if c["clause_id"] == clause_id:
                return c
        return None

    # Helper to duplicate and modify a clause
    def duplicate_and_modify(clause_id, new_content, amendment_ref, transitional_rule):
        base_c = find_clause(clause_id)
        if base_c:
            # Update base clause validity
            base_c["effective_to"] = "2026-03-01"
            base_c["transitional_rule"] = transitional_rule
            
            # Create amended version
            amended_c = base_c.copy()
            amended_c["version"] = "Amendment No. 2026-01"
            amended_c["content"] = new_content
            amended_c["effective_from"] = "2026-03-01"
            amended_c["effective_to"] = None
            amended_c["transitional_rule"] = transitional_rule
            amended_c["amendment_ref"] = amendment_ref
            amended_c["source_order"] = base_c["source_order"] + 1000  # put later
            modified_clauses.append(amended_c)

    # 1. Earnings disregard §6.4.1
    base_641 = find_clause("§6.4.1")
    if base_641:
        new_content_641 = base_641["content"].replace("$120 per month", "$175 per month")
        duplicate_and_modify("§6.4.1", new_content_641, "Amendment §1.1", "§5.1")

    # 2. Reporting changes §4.3.2
    base_432 = find_clause("§4.3.2")
    if base_432:
        new_content_432 = base_432["content"].replace("10 calendar days", "14 calendar days")
        duplicate_and_modify("§4.3.2", new_content_432, "Amendment §2.1", "§5.2")

    # 3. Overpayments reporting timeframe §9.1.4
    base_914 = find_clause("§9.1.4")
    if base_914:
        new_content_914 = base_914["content"].replace("30 calendar days", "14 calendar days")
        duplicate_and_modify("§9.1.4", new_content_914, "Amendment §2.2", "§5.2")

    # 4. Income thresholds table §6.6.1
    base_661 = find_clause("§6.6.1")
    if base_661:
        new_content_661 = (base_661["content"]
                           .replace("$1,180", "$1,225")
                           .replace("$1,590", "$1,650")
                           .replace("$2,000", "$2,075")
                           .replace("$2,410", "$2,500")
                           .replace("$2,820", "$2,925")
                           .replace("$410", "$425"))
        duplicate_and_modify("§6.6.1", new_content_661, "Amendment §3.1", "§5.1")

    # 5. Sanctions §10.5.2
    base_1052 = find_clause("§10.5.2")
    if base_1052:
        new_content_1052 = base_1052["content"].replace("20 per cent", "15 per cent")
        duplicate_and_modify("§10.5.2", new_content_1052, "Amendment §4.1", "§5.1")

    # 6. Insert new §10.5.3A
    new_1053A = {
        "clause_id": "§10.5.3A",
        "clause_title": "Sanctions (Amended)",
        "content": "A sanction must not be imposed in respect of a failure to report where the change of circumstances in question would have increased the award.",
        "part_title": "Part 10 — Suspension, Termination and Sanctions",
        "section_title": "10.5 Sanctions",
        "version": "Amendment No. 2026-01",
        "source_order": len(clauses) + len(amendment_clauses) + 1,
        "effective_from": "2026-03-01",
        "effective_to": None,
        "transitional_rule": "§5.1",
        "amendment_ref": "Amendment §4.2"
    }
    
    # Combine original, modified, amendment clauses, and new clause
    result_clauses = clauses + modified_clauses + amendment_clauses + [new_1053A]
    return result_clauses

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_policy_manual> [path_to_amendment]")
        sys.exit(1)

    file_path = sys.argv[1]
    
    # Auto-detect amendment path if not provided
    amendment_path = sys.argv[2] if len(sys.argv) > 2 else "1 - The Grounded Answer/Amendment No. 2026-01.md"
    
    print(f"Parsing policy manual: {file_path}")
    clauses, num_sections, version = parse_policy_manual(file_path)
    
    # Apply amendment layer
    clauses = apply_amendment_layer(clauses, amendment_path)
    
    output_path = "clauses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=4, ensure_ascii=False)
        
    print("\n--- Parsing Ingestion Summary ---")
    print(f"File processed: {file_path}")
    print(f"Version/Quarter detected: {version}")
    print(f"Number of sections: {num_sections}")
    print(f"Total number of versioned clauses: {len(clauses)}")
    print(f"Data saved to: {output_path}")
    print("---------------------------------")

if __name__ == "__main__":
    main()
