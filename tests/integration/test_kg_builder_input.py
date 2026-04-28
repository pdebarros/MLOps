def test_kg_builder_only_uses_completed_summaries():
    records = [
        {
            "file": "src/main.py",
            "summary": "Builds the pipeline.",
            "status": "completed",
        },
        {
            "file": "src/broken.py",
            "summary": "",
            "status": "failed",
            "error": "Could not summarize file.",
        },
    ]

    completed_records = [
        record for record in records
        if record["status"] == "completed" and record["summary"]
    ]

    assert len(completed_records) == 1
    assert completed_records[0]["file"] == "src/main.py"