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

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_policy_manual>")
        sys.exit(1)

    file_path = sys.argv[1]
    
    print(f"Parsing policy manual: {file_path}")
    clauses, num_sections, version = parse_policy_manual(file_path)
    
    output_path = "clauses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=4, ensure_ascii=False)
        
    print("\n--- Parsing Ingestion Summary ---")
    print(f"File processed: {file_path}")
    print(f"Version/Quarter detected: {version}")
    print(f"Number of sections: {num_sections}")
    print(f"Number of clauses: {len(clauses)}")
    print(f"Data saved to: {output_path}")
    print("---------------------------------")

if __name__ == "__main__":
    main()
