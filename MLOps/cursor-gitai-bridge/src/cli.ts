#!/usr/bin/env node
/**
 * Commands:
 *   watch [--file PATH] [--out-dir DIR] [--from-start]
 *   ask [--cwd DIR] <question>
 *   session [--cwd DIR] [--new | --agent-id ID] [--data-dir DIR] <question>
 */
import fs from "node:fs/promises";
import path from "node:path";
import {
  CursorAgentError,
  promptOnce,
  sessionNew,
  sessionResume,
} from "./cursorAgent.js";
import { runWatch } from "./watchGitAi.js";

const LAST_AGENT_FILE = "last-agent.json";

async function saveLastAgent(dataDir: string, agentId: string): Promise<void> {
  await fs.mkdir(dataDir, { recursive: true });
  const p = path.join(dataDir, LAST_AGENT_FILE);
  await fs.writeFile(p, JSON.stringify({ agentId, saved_at: new Date().toISOString() }, null, 2), "utf8");
}

async function readLastAgent(dataDir: string): Promise<string | null> {
  try {
    const raw = await fs.readFile(path.join(dataDir, LAST_AGENT_FILE), "utf8");
    const j = JSON.parse(raw) as { agentId?: string };
    return j.agentId ?? null;
  } catch {
    return null;
  }
}

function takeQuestion(argv: string[], i: number): string {
  const rest = argv.slice(i).join(" ").trim();
  if (!rest) {
    throw new Error("missing question text");
  }
  return rest;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const cmd = argv[0];

  if (!cmd || cmd === "-h" || cmd === "--help") {
    console.log(`
cursor-gitai-bridge — tail git-ai NDJSON + Cursor Agent SDK

Usage:
  CURSOR_API_KEY=... node dist/cli.js watch [--file PATH] [--out-dir data] [--from-start]
  CURSOR_API_KEY=... node dist/cli.js ask [--cwd DIR] <question>
  CURSOR_API_KEY=... node dist/cli.js session [--cwd DIR] [--agent-id ID] [--data-dir DIR] <question>

Environment:
  CURSOR_API_KEY   Required for ask/session (https://cursor.com/dashboard/cloud-agents)

Examples:
  npm run dev -- watch --file ../gitai/cursor-proposed-changes.ndjson
  npm run dev -- ask --cwd .. "Summarize the gitai folder"
  npm run dev -- session --cwd .. "What does tracker_worker.py do?"
  npm run dev -- session --cwd .. --agent-id bc-xxxx "Follow-up question"
  npm run dev -- session --cwd .. --new "Start a fresh agent"
`);
    process.exit(cmd ? 0 : 1);
  }

  if (cmd === "watch") {
    await runWatch(argv.slice(1));
    return;
  }

  let cwd = process.cwd();
  let agentId: string | undefined;
  let dataDir = "data";
  let forceNew = false;
  let i = 1;
  if (cmd === "ask" || cmd === "session") {
    while (i < argv.length) {
      const a = argv[i];
      if (a === "--cwd" && argv[i + 1]) {
        cwd = path.resolve(argv[++i]!);
        i++;
      } else if (a === "--agent-id" && argv[i + 1]) {
        agentId = argv[++i]!;
        i++;
      } else if (a === "--data-dir" && argv[i + 1]) {
        dataDir = argv[++i]!;
        i++;
      } else if (a === "--new") {
        forceNew = true;
        i++;
      } else {
        break;
      }
    }
  }

  const question = takeQuestion(argv, i);

  if (cmd === "ask") {
    try {
      const result = await promptOnce(cwd, question);
      console.log(result.status);
      if (result.result) {
        console.log(result.result);
      }
    } catch (e) {
      if (e instanceof CursorAgentError) {
        console.error("CursorAgentError:", e.message);
        process.exit(1);
      }
      throw e;
    }
    return;
  }

  if (cmd === "session") {
    try {
      const resumeId =
        agentId ?? (!forceNew ? await readLastAgent(dataDir) : null);
      if (resumeId && !forceNew) {
        console.error(`[session] resuming agent ${resumeId}`);
        await sessionResume(cwd, resumeId, question);
      } else {
        console.error(forceNew ? "[session] new agent (--new)" : "[session] new agent");
        const { agentId: newId } = await sessionNew(cwd, question);
        if (newId) {
          await saveLastAgent(dataDir, newId);
          console.error(`[session] agent id saved → ${path.join(dataDir, LAST_AGENT_FILE)}`);
          console.error(`[session] export CURSOR_AGENT_ID=${newId}`);
        }
      }
    } catch (e) {
      if (e instanceof CursorAgentError) {
        console.error("CursorAgentError:", e.message);
        process.exit(1);
      }
      throw e;
    }
    return;
  }

  console.error("unknown command:", cmd);
  process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
