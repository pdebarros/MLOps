from pathlib import Path

def test_no_hardcoded_secrets_in_python_files():
    suspicious_terms = [
        "NEO4J_PASSWORD=",
        "GOOGLE_API_KEY=",
        "GITHUB_TOKEN=",
        "HF_TOKEN=",
    ]

    python_files = list(Path("KG_agent").rglob("*.py"))

    for file_path in python_files:
        text = file_path.read_text(errors="ignore")

        for term in suspicious_terms:
            assert term not in text, f"Possible hardcoded secret in {file_path}"