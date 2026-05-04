/**
 * Cursor Cloud Agents SDK — local runtime against a repo checkout.
 * Requires CURSOR_API_KEY (https://cursor.com/dashboard/cloud-agents).
 */
import { Agent, CursorAgentError } from "@cursor/sdk";
import type { Run } from "@cursor/sdk";

export function getApiKey(): string {
  const k = process.env.CURSOR_API_KEY?.trim();
  if (!k) {
    throw new Error(
      "CURSOR_API_KEY is not set. Create a key at https://cursor.com/dashboard/cloud-agents",
    );
  }
  return k;
}

export async function promptOnce(cwd: string, question: string) {
  return Agent.prompt(question, {
    apiKey: getApiKey(),
    model: { id: "composer-2" },
    local: { cwd },
  });
}

async function streamRun(run: Run) {
  for await (const event of run.stream()) {
    if (event.type === "assistant") {
      for (const block of event.message.content) {
        if (block.type === "text") {
          process.stdout.write(block.text);
        }
      }
    }
  }
  process.stdout.write("\n");
  return run.wait();
}

export async function sessionNew(cwd: string, question: string) {
  const agent = await Agent.create({
    apiKey: getApiKey(),
    model: { id: "composer-2" },
    local: { cwd },
  });
  try {
    const run = await agent.send(question);
    const result = await streamRun(run);
    const id = agent.agentId;
    return { agentId: id, result };
  } finally {
    await agent[Symbol.asyncDispose]();
  }
}

export async function sessionResume(cwd: string, agentId: string, question: string) {
  const agent = await Agent.resume(agentId, {
    apiKey: getApiKey(),
    model: { id: "composer-2" },
    local: { cwd },
  });
  try {
    const run = await agent.send(question);
    const result = await streamRun(run);
    return { agentId, result };
  } finally {
    await agent[Symbol.asyncDispose]();
  }
}

export { CursorAgentError };
