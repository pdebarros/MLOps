#for testing

from google.adk.agents.llm_agent import Agent, ParallelAgent
from google.adk.tools import ToolContext 
from tools import save_code_files



subagent_1 = Agent(
    model='gemini-2.5-flash',
    name='Subagent 1',
    description='A summarizer agent that summarizes the code files and stores them in the session state.',
    instruction='Answer user questions to the best of your knowledge',
    sub_agents=[],
    tools=[save_code_files]
)


subagent_2 = Agent(
    model='gemini-2.5-flash',
    name='Subagent 2',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    sub_agents=[],
    tools=[save_code_files]
)

subagent_3 = Agent(
    model='gemini-2.5-flash',
    name='Subagent 3',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    sub_agents=[],
    tools=[save_code_files]
)

sub_orchestrator = ParallelAgent(
    model='gemini-2.5-flash',
    name='Sub Orchestrator',
    description='A parallel agent that creates summaries for knowledge graph creation in parallel.',
    instruction='Orchestrate the subagents to complete the task.',
    sub_agents=[subagent_1, subagent_2, subagent_3],
    tools=[save_code_files]
)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='Orchestrator',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    sub_agents=[sub_orchestrator],
    tools=[save_code_files] # will also have run_LLM_transformer tool
)
