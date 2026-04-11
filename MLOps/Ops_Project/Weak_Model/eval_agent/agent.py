from google.adk.agents.llm_agent import Agent

from .prompt import SYSTEM_INSTRUCTION
from .tools import (
    fetch_neighborhood_from_neo4j,
    list_entity_id_samples,
    query_gemini_kg_student,
    query_weak_kg_student,
    retrieve_from_rag_corpus,
    save_scores_to_gcs,
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description=(
        "Evaluates a Cypher-backed KG student (Gemini Flash multi-step explorer by default; optional Groq "
        "Llama baseline) by posing questions, grounding with RAG + subgraphs, and scoring QA on a 0.0–1.0 scale."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        list_entity_id_samples,
        fetch_neighborhood_from_neo4j,
        retrieve_from_rag_corpus,
        query_gemini_kg_student,
        query_weak_kg_student,
        save_scores_to_gcs,
    ],
)
