# import asyncio
# import logging
# import uuid
# from typing import List
# from google.adk import Workflow, Agent, AgentTool, Task
# from google.adk.runtime import LocalRunner
# from google.cloud import storage  # Ensure this is google-cloud-storage
# # --- STEP 1: Logging Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(task_id)s] %(message)s'
# )
# logger = logging.getLogger("Orchestrator")

# def get_logger_adapter(task_id: str):
#     return logging.LoggerAdapter(logger, {"task_id": task_id})

# # --- STEP 2: The Specialist (Sub-Agent) & Tool ---

# summarizer_agent = Agent(
#     name="Code-Summarizer",
#     instructions="Extract classes, dependencies, and logic flow from Python code."
# )

# async def run_summarizer_tool(file_name: str, file_content: str):
#     """
#     Agent-based tool. Arguments 'file_name' and 'file_content' 
#     must match the keys in the input dictionary.
#     """
#     task_id = str(uuid.uuid4())[:8]
#     log = get_logger_adapter(task_id)
    
#     log.info(f"START: Analyzing '{file_name}'")
    
#     try:
#         # Pass the unpacked text to the sub-agent
#         response = await summarizer_agent.run_async(f"Analyze this code:\n\n{file_content}")
        
#         log.info(f"SUCCESS: Summary generated for '{file_name}'")
#         return {"file": file_name, "summary": response.text, "status": "completed"}
        
#     except Exception as e:
#         log.error(f"FAILURE: '{file_name}' failed. Error: {str(e)}")
#         return {"file": file_name, "summary": None, "status": "failed", "error": str(e)}

# summarize_tool = AgentTool(
#     name="summarize_python_code",
#     fn=run_summarizer_tool,
#     description="Analyzes code and returns a structural summary."
# )

# # --- STEP 3: The Master (Orchestrator) ---

# master_orchestrator = Agent(
#     name="Master-Orchestrator",
#     instructions="Review all summaries and build the final Knowledge Graph.",
#     tools=[summarize_tool]
# )

# # --- STEP 4: GCS Data Loading Logic ---

# def load_code_files(bucket_name: str = "codebases-03-26", prefix: str = None):
#     """
#     Lists python files, downloads them as text, and prepares them for the ADK.
#     """
#     file_data_list = []
#     try:
#         client = storage.Client()
#         bucket = client.get_bucket(bucket_name)
#         blobs = bucket.list_blobs(prefix=prefix)
        
#         extensions = ('.py',)
        
#         for blob in blobs:
#             if blob.name.endswith(extensions):
#                 # Unpack: Download bytes and decode to UTF-8 string
#                 content_text = blob.download_as_bytes().decode('utf-8')
                
#                 # CRITICAL: Keys here must match run_summarizer_tool arguments
#                 file_data_list.append({
#                     "file_name": blob.name, 
#                     "file_content": content_text
#                 })
#                 logger.info(f"Loaded and unpacked: {blob.name}")
                
#     except Exception as e:
#         logger.error(f"Error loading GCS files: {e}")
#         return []

#     return file_data_list

# # --- STEP 5: The Workflow Factory ---

# def create_pipeline(data_list: List[dict]):
#     wf = Workflow(name="Logged-Parallel-Pipeline")
    
#     # Fan-out: Parallel tasks expect a list of dicts. 
#     # Each dict is unpacked as kwargs into the tool.
#     summarize_tasks = wf.add_parallel_tasks(
#         agent=master_orchestrator,
#         tool="summarize_python_code",
#         inputs=data_list,
#         name="ParallelSummarization"
#     )

#     wf.add_task(
#         agent=master_orchestrator,
#         tool="build_final_knowledge_graph",
#         args={"results": summarize_tasks},
#         dependencies=summarize_tasks,
#         name="FinalGraphConstruction"
#     )

#     return wf

# # --- STEP 6: Execution ---

# async def main():
#     # 1. Fetch and unpack data from GCS
#     bucket = "codebases-03-26"
#     logger.info(f"Fetching source files from {bucket}...")
#     files = load_code_files(bucket_name=bucket)
    
#     if not files:
#         logger.warning("No files found to process. Exiting.")
#         return

#     # 2. Build and run the pipeline
#     pipeline = create_pipeline(files)
#     runner = LocalRunner()
    
#     logger.info("Starting Parallel ADK Runner...")
#     result = await runner.run(pipeline)
#     logger.info(f"Final Result: {result}")

# if __name__ == "__main__":
#     asyncio.run(main())