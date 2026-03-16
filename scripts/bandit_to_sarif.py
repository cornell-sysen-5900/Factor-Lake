import sys
import json
import os
import hashlib

def generate_fingerprint(file_path, line_number, test_id):
    data = f"{file_path}:{line_number}:{test_id}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def convert_bandit_to_sarif(bandit_json_path, sarif_output_path):
    with open(bandit_json_path, "r") as f:
        bandit_data = json.load(f)

    sarif = {
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "Bandit",
                    "informationUri": "https://bandit.readthedocs.io/",
                    "rules": []
                }
            },
            "results": []
        }]
    }

    rules_seen = {}
    #results = []

    for result in bandit_data.get("results", []):
        test_id = result.get("test_id", "UNKNOWN")
        rule_id = f"BANDIT.{test_id}"
        description = result.get("issue_text", "")
        file_path = os.path.relpath(result.get("filename", ""))
        start_line = result.get("line_number", 1)

        if rule_id not in rules_seen:
            sarif["runs"][0]["tool"]["driver"]["rules"].append({
                "id": rule_id,
                "name": result.get("test_name", test_id),
                "shortDescription": {"text": description},
                "fullDescription": {"text": description},
                "helpUri": result.get("more_info", ""),
                "properties": {
                    "tags": ["security", "bandit"],
                    "precision": "very-high"
                }
            })
            rules_seen[rule_id] = True

        sarif["runs"][0]["results"].append({
            "ruleId": rule_id,
            "message": {"text": description},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": file_path},
                    "region": {"startLine": start_line}
                }
            }],
            "fingerprints": {
                "primary": generate_fingerprint(file_path, start_line, test_id)
            }
        })

    with open(sarif_output_path, "w") as f:
        json.dump(sarif, f, indent=2)
    print(f"✅ Converted Bandit JSON to SARIF at {sarif_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bandit_to_sarif.py <input_bandit_json> <output_sarif>")
        sys.exit(1)

    convert_bandit_to_sarif(sys.argv[1], sys.argv[2])