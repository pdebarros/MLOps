def test_summary_records_schema():
    records = [
        {
            "file": "src/main.py",
            "summary": "Main pipeline file.",
            "status": "completed",
        },
        {
            "file": "src/helper.py",
            "summary": "Helper functions.",
            "status": "completed",
        },
    ]

    for record in records:
        assert "file" in record
        assert "summary" in record
        assert "status" in record
        assert record["status"] in ["completed", "failed"]
        assert record["file"].endswith(".py")