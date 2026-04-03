import json
import logging
from typing import Any

import torch
from neo4j import GraphDatabase
from transformers import pipeline

from config import Config

logger = logging.getLogger("Orchestrator.KG-Tool")

MODEL_ID = Config.HF_KG_MODEL
_kg_generator: Any = None


def _get_kg_generator():
    """Load HF pipeline on first use (avoids download at import time)."""
    global _kg_generator
    if _kg_generator is None:
        device = 0 if torch.cuda.is_available() else -1
        _kg_generator = pipeline(
            "text-generation", model=MODEL_ID, device=device
        )
    return _kg_generator

# --- STEP 2: Neo4j Connection Logic ---
NEO4J_URI = Config.NEO4J_URI
NEO4J_USER = Config.NEO4J_USER
NEO4J_PASSWORD = Config.NEO4J_PASSWORD

def save_to_neo4j(graph_json: str):
    """Parses JSON and commits to Neo4j using Cypher."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        data = json.loads(graph_json)
        with driver.session() as session:
            # 1. Create Nodes
            for node in data.get('nodes', []):
                session.run(
                    "MERGE (n:CodeEntity {id: $id}) SET n.type = $type",
                    id=node['id'], type=node['type']
                )
            
            # 2. Create Edges
            for edge in data.get('edges', []):
                session.run(
                    """
                    MATCH (a:CodeEntity {id: $source})
                    MATCH (b:CodeEntity {id: $target})
                    MERGE (a)-[r:RELATION {type: $rel}]->(b)
                    """,
                    source=edge['source'], target=edge['target'], rel=edge['relation']
                )
        return True
    except Exception as e:
        logger.error(f"Neo4j Commit Failed: {e}")
        return False
    finally:
        driver.close()

# --- STEP 3: The Combined Tool Function ---

def build_kg_and_push_to_neo4j(results: list):
    """Master tool: Summary List -> HF Model -> Neo4j."""
    
    successful_summaries = [
        r["summary"]
        for r in results
        if r.get("status") == "completed" and r.get("summary")
    ]
    if not successful_summaries:
        return {
            "status": "error",
            "message": "No successful summaries to build a knowledge graph from.",
        }
    combined_context = "\n---\n".join(successful_summaries)
    
    prompt = (
        "Instructions: Convert code summaries into a Knowledge Graph JSON. "
        "Include 'nodes' [id, type] and 'edges' [source, target, relation].\n\n"
        f"Context:\n{combined_context}\n\nJSON:"
    )
    
    try:
        # Inference
        output = _get_kg_generator()(prompt, max_new_tokens=1024, temperature=0.1)
        graph_json = output[0]['generated_text']
        
        # Persistence
        success = save_to_neo4j(graph_json)
        
        return {
            "status": "success" if success else "neo4j_error",
            "files_processed": len(successful_summaries),
            "raw_graph": graph_json
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}