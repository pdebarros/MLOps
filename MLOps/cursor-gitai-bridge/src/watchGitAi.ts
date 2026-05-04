/**
 * Tail ``cursor-proposed-changes.ndjson`` (same format as ``track_cursor_gitai.py`` output).
 */
import fs from "node:fs";
import path from "node:path";
import { appendJsonl } from "./store.js";

function parseArgs(argv: string[]): { file: string; outDir: string; fromStart: boolean } {
  let file = "cursor-proposed-changes.ndjson";
  let outDir = "data";
  let fromStart = false;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--file" && argv[i + 1]) {
      file = argv[++i]!;
    } else if (a === "--out-dir" && argv[i + 1]) {
      outDir = argv[++i]!;
    } else if (a === "--from-start") {
      fromStart = true;
    }
  }
  return { file, outDir, fromStart };
}

export async function runWatch(argv: string[]): Promise<void> {
  const { file, outDir, fromStart } = parseArgs(argv);
  const abs = path.resolve(file);
  const outName = "git-ai-prompts.jsonl";

  let pos = 0;
  if (!fromStart) {
    try {
      pos = fs.statSync(abs).size;
    } catch {
      pos = 0;
    }
  }

  console.error(`[watch] tailing ${abs} → ${path.join(outDir, outName)} (from byte ${pos})`);

  const poll = async () => {
    try {
      const st = fs.statSync(abs);
      if (st.size < pos) {
        pos = 0;
      }
      if (st.size <= pos) {
        return;
      }
      const fd = fs.openSync(abs, "r");
      try {
        const len = st.size - pos;
        const buf = Buffer.alloc(len);
        fs.readSync(fd, buf, 0, len, pos);
        pos = st.size;
        const text = buf.toString("utf8");
        const lines = text.split("\n");
        for (const line of lines) {
          const s = line.trim();
          if (!s) {
            continue;
          }
          try {
            const rec = JSON.parse(s) as Record<string, unknown>;
            await appendJsonl(outDir, outName, {
              ingested_at: new Date().toISOString(),
              source_file: abs,
              record: rec,
            });
            const pid = rec.prompt_id;
            console.error(`[watch] recorded prompt_id=${pid ?? "?"}`);
          } catch {
            await appendJsonl(outDir, outName, {
              ingested_at: new Date().toISOString(),
              source_file: abs,
              parse_error: true,
              raw_line: s.slice(0, 500),
            });
          }
        }
      } finally {
        fs.closeSync(fd);
      }
    } catch (e) {
      const err = e as NodeJS.ErrnoException;
      if (err.code === "ENOENT") {
        return;
      }
      console.error("[watch] read error:", err.message);
    }
  };

  await poll();
  const iv = setInterval(() => {
    void poll();
  }, 600);

  process.on("SIGINT", () => {
    clearInterval(iv);
    process.exit(0);
  });
}
