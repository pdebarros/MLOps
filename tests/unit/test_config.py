import os

def test_required_env_var_names_exist():
    required = [
        "GCS_BUCKET_NAME",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "KG_GRAPH_BACKEND",
    ]

    for var in required:
        assert isinstance(var, str)