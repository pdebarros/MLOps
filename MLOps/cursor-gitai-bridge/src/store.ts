import fs from "node:fs/promises";
import path from "node:path";

/** Append one JSON object per line (JSONL). */
export async function appendJsonl(dir: string, filename: string, record: unknown): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
  const p = path.join(dir, filename);
  await fs.appendFile(p, JSON.stringify(record) + "\n", "utf8");
}
