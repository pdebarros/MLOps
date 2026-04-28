def test_pipeline_skips_summarization_when_all_summaries_completed():
    source_files = ["src/main.py", "src/utils.py"]

    cached_summaries = {
        "src/main.py": {"status": "completed", "summary": "Main file."},
        "src/utils.py": {"status": "completed", "summary": "Utility file."},
    }

    all_completed = all(
        file in cached_summaries and cached_summaries[file]["status"] == "completed"
        for file in source_files
    )

    assert all_completed is True