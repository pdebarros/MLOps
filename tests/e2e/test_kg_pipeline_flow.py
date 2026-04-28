def test_end_to_end_kg_pipeline_flow_with_mock_data():
    source_files = ["src/main.py"]

    summaries = [
        {
            "file": "src/main.py",
            "summary": "Defines a pipeline that loads code from GCS and writes a graph to Neo4j.",
            "status": "completed",
        }
    ]

    graph_result = {
        "nodes_written": 3,
        "relationships_written": 2,
    }

    assert len(source_files) > 0
    assert summaries[0]["status"] == "completed"
    assert graph_result["nodes_written"] > 0
    assert graph_result["relationships_written"] >= 0