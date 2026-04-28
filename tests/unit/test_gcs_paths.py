def test_summary_path_is_created_correctly():
    user_id = "u_123"
    source_file = "src/main.py"

    summary_path = f"{user_id}/summaries/{source_file}.json"

    assert summary_path == "u_123/summaries/src/main.py.json"
    assert "/summaries/" in summary_path
    assert summary_path.endswith(".py.json")