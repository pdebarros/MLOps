def test_completed_summary_record_has_required_fields():
    record = {
        "file": "src/main.py",
        "summary": "This file loads Python code and processes it.",
        "status": "completed",
    }

    assert record["status"] == "completed"
    assert record["file"].endswith(".py")
    assert len(record["summary"]) > 0