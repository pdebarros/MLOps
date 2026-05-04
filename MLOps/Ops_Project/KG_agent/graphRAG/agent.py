"""GraphRAG ADK agent — local hybrid retrieval over the experience KG."""
from __future__ import annotations

from google.adk.agents.llm_agent import Agent

from .config import Config
from .prompt import SYSTEM_INSTRUCTION
from .tools import (
    assemble_context,
    assess_query_difficulty,
    fetch_neighborhood,
    rag_corpus_query,
    score_neighborhood,
    score_rag_evidence,
    set_source_weights,
    synthesize_evidence,
    vector_seed_search,
)


root_agent = Agent(
    model=Config.AGENT_MODEL,
    name="graphRAG_agent",
    description=(
        "Two-source GraphRAG agent that answers questions about a user's "
        "programming experience by combining (1) the Neo4j experience graph "
        "via vector seed search + neighborhood expansion, and (2) the user's "
        "Vertex AI RAG Engine corpus of GCS chunks. The agent weights both "
        "sources based on its confidence and synthesizes a grounded answer."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        assess_query_difficulty,
        vector_seed_search,
        fetch_neighborhood,
        score_neighborhood,
        rag_corpus_query,
        score_rag_evidence,
        set_source_weights,
        synthesize_evidence,
        assemble_context,
    ],
)
