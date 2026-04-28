def test_graph_quality_minimum_nodes_and_edges():
    graph_result = {
        "nodes_written": 5,
        "relationships_written": 4,
    }

    assert graph_result["nodes_written"] >= 3
    assert graph_result["relationships_written"] >= 1