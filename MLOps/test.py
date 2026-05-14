import vertexai
from vertexai.preview import rag

# Configuration
PROJECT_ID = "project-b0cc8bfd-0a29-4ee2-9a9"
LOCATION = "europe-west4" 

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

def initialize_spanner_corpus():
    # The publisher_model string must follow a specific pattern.
    # For text-embedding-004, use the shorter global publisher path 
    # which the SDK resolves to your local region automatically.
    emb_config = rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-004"
    )

    try:
        print(f"Creating Spanner-backed RAG corpus in {LOCATION}...")
        # Note: This may take a moment to provision the underlying Spanner instance
        new_corpus = rag.create_corpus(
            display_name="netherlands_spanner_rag",
            embedding_model_config=emb_config
        )
        print(f"Successfully initialized corpus: {new_corpus.name}")
    except Exception as e:
        # If it still fails, the error message will help identify 
        # if it's a path issue or a regional availability issue.
        print(f"Initialization failed: {e}")

if __name__ == "__main__":
    initialize_spanner_corpus()