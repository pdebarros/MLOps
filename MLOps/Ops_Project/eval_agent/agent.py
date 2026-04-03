from google.adk.agents.llm_agent import Agent

from .prompt import SYSTEM_INSTRUCTION
from .tools import (
    fetch_neighborhood_from_neo4j,
    list_entity_id_samples,
    retrieve_from_rag_corpus,
    save_scores_to_gcs,
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description=(
        "Evaluates a Neo4j knowledge graph against a Vertex RAG corpus by sampling "
        "neighborhoods and scoring alignment on a 0.0–1.0 scale."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        list_entity_id_samples,
        fetch_neighborhood_from_neo4j,
        retrieve_from_rag_corpus,
        save_scores_to_gcs,
    ],
)
