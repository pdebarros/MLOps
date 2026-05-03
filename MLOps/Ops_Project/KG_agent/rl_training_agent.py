"""
RL orchestration for KG + RAG-aware code-improvement training.

What this module does:
1) Runs pipeline3 for multiple user/project scenarios (including mixed complexity users).
2) Uploads all Python files for each user into Vertex RAG Engine (display name = user id).
3) Ingests "10 Cursor agents" outputs (stored as JSONL rollouts) for one shared prompt.
4) Computes a GRPO-style reward using:
   - group-relative quality (syntax + KG/RAG grounding hints),
   - inter-output diversity.
5) Applies a LoRA update to a base causal LM using weighted supervised loss
   (weights derived from GRPO advantages).

The implementation is intentionally pragmatic and lightweight:
- It does not launch Cursor agents itself; it trains from saved outputs.
- It uses optional imports for Vertex RAG and PEFT/Transformers training.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from pipeline3 import run_pipeline3

logger = logging.getLogger("Orchestrator.RLTrain")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _list_python_files(source_dir: Path) -> list[Path]:
    return sorted([p for p in source_dir.rglob("*.py") if p.is_file()])


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower() or "unknown"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text))


def _jaccard_distance(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return 1.0 - (inter / union if union else 0.0)


def _syntax_score_python(code_text: str) -> float:
    import ast

    try:
        ast.parse(code_text)
        return 1.0
    except Exception:
        return 0.0


@dataclass
class ProjectScenario:
    user_id: str
    neo4j_database: str
    description: str
    # Optional metadata to track "combined" project cases.
    project_roots: list[str]


@dataclass
class CursorRollout:
    user_id: str
    neo4j_database: str
    rag_corpus: str
    prompt: str
    agent_id: str
    changed_code: str
    metadata: dict[str, Any]


@dataclass
class RewardRecord:
    prompt: str
    agent_id: str
    reward: float
    diversity: float
    quality: float
    advantage: float
    user_id: str
    neo4j_database: str
    rag_corpus: str


class RAGEngineUploader:
    """Uploads python files into Vertex RAG corpus; display name convention tracks user id."""

    def __init__(self, corpus_resource: str):
        self.corpus_resource = corpus_resource

    def upload_user_docs(self, user_id: str, source_dir: Path) -> dict[str, Any]:
        py_files = _list_python_files(source_dir)
        if not py_files:
            return {
                "status": "skipped",
                "reason": "no_python_files",
                "user_id": user_id,
                "source_dir": str(source_dir),
            }

        try:
            import vertexai
            from vertexai import rag
        except Exception as e:
            return {
                "status": "error",
                "reason": f"vertexai_rag_import_failed: {e}",
                "user_id": user_id,
                "count": len(py_files),
            }

        # vertexai.init project/location can be inferred from ADC + resource name in most setups.
        # We still call init() with no args for consistency.
        vertexai.init()
        file_paths = [str(p.resolve()) for p in py_files]
        display_name = f"user-{_safe_slug(user_id)}"

        # Vertex RAG APIs vary by SDK version; try common call shapes.
        try:
            # Newer SDKs commonly expose import_files.
            result = rag.import_files(
                corpus_name=self.corpus_resource,
                paths=file_paths,
                chunk_size=1024,
                chunk_overlap=128,
                display_name=display_name,
            )
            return {
                "status": "uploaded",
                "user_id": user_id,
                "python_files": len(file_paths),
                "display_name": display_name,
                "result": str(result),
            }
        except TypeError:
            pass
        except Exception as e:
            return {
                "status": "error",
                "reason": f"rag.import_files_failed: {e}",
                "user_id": user_id,
                "python_files": len(file_paths),
            }

        try:
            # Fallback call shape seen in some SDK versions.
            result = rag.import_files(
                self.corpus_resource,
                file_paths,
            )
            return {
                "status": "uploaded",
                "user_id": user_id,
                "python_files": len(file_paths),
                "display_name": display_name,
                "result": str(result),
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"rag_import_fallback_failed: {e}",
                "user_id": user_id,
                "python_files": len(file_paths),
            }


class GRPOReward:
    """
    GRPO-style reward:
    - quality_i = weighted sum of syntax + grounding overlap
    - diversity_i = mean distance to peers in same prompt group
    - reward_i = alpha * quality_i + beta * diversity_i
    - advantage_i = reward_i - mean(reward_group)
    """

    def __init__(self, alpha_quality: float = 0.7, beta_diversity: float = 0.3):
        self.alpha_quality = alpha_quality
        self.beta_diversity = beta_diversity

    @staticmethod
    def _grounding_overlap(changed_code: str, prompt: str, rag_corpus: str, neo4j_database: str) -> float:
        # Proxy metric: encourage lexical overlap with prompt + graph/rag identifiers.
        base = _tokenize(changed_code)
        refs = _tokenize(prompt) | _tokenize(rag_corpus) | _tokenize(neo4j_database)
        if not refs:
            return 0.0
        return len(base & refs) / len(refs)

    def score_group(self, rollouts: list[CursorRollout]) -> list[RewardRecord]:
        if not rollouts:
            return []

        rewards: list[tuple[CursorRollout, float, float, float]] = []
        for i, r in enumerate(rollouts):
            syntax = _syntax_score_python(r.changed_code)
            grounding = self._grounding_overlap(
                changed_code=r.changed_code,
                prompt=r.prompt,
                rag_corpus=r.rag_corpus,
                neo4j_database=r.neo4j_database,
            )
            quality = 0.6 * syntax + 0.4 * grounding

            peer_distances: list[float] = []
            for j, other in enumerate(rollouts):
                if i == j:
                    continue
                peer_distances.append(_jaccard_distance(r.changed_code, other.changed_code))
            diversity = sum(peer_distances) / len(peer_distances) if peer_distances else 0.0

            reward = self.alpha_quality * quality + self.beta_diversity * diversity
            rewards.append((r, reward, diversity, quality))

        group_mean = sum(x[1] for x in rewards) / len(rewards)
        out: list[RewardRecord] = []
        for r, reward, diversity, quality in rewards:
            out.append(
                RewardRecord(
                    prompt=r.prompt,
                    agent_id=r.agent_id,
                    reward=reward,
                    diversity=diversity,
                    quality=quality,
                    advantage=(reward - group_mean),
                    user_id=r.user_id,
                    neo4j_database=r.neo4j_database,
                    rag_corpus=r.rag_corpus,
                )
            )
        return out


class LoRAUpdater:
    """LoRA fine-tuning with weighted supervised objective from GRPO advantages."""

    def __init__(self, base_model: str, output_dir: Path):
        self.base_model = base_model
        self.output_dir = output_dir

    def fit(self, rollouts: list[CursorRollout], rewards: list[RewardRecord], epochs: int = 1) -> dict[str, Any]:
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            return {"status": "error", "reason": f"lora_dependencies_missing: {e}"}

        # Map rewards by agent_id for weighting.
        reward_by_agent = {r.agent_id: r for r in rewards}
        samples = [r for r in rollouts if r.agent_id in reward_by_agent]
        if not samples:
            return {"status": "error", "reason": "no_samples_with_rewards"}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        step = 0
        losses: list[float] = []

        for _ in range(max(1, epochs)):
            random.shuffle(samples)
            for s in samples:
                rr = reward_by_agent[s.agent_id]
                # Convert group-relative advantage to positive sample weight.
                weight = 1.0 / (1.0 + math.exp(-4.0 * rr.advantage))
                prompt = f"Project prompt:\n{s.prompt}\n\nCode changes:\n{s.changed_code}"
                toks = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(device)
                out = model(**toks, labels=toks["input_ids"])
                loss = out.loss * float(weight)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.detach().cpu().item()))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        return {
            "status": "ok",
            "steps": step,
            "loss_mean": (sum(losses) / len(losses)) if losses else None,
            "output_dir": str(self.output_dir),
        }


class KGRAGRlAgent:
    def __init__(
        self,
        rag_corpus_resource: str,
        base_model: str,
        lora_output_dir: Path,
    ):
        self.rag_uploader = RAGEngineUploader(corpus_resource=rag_corpus_resource)
        self.rewarder = GRPOReward(alpha_quality=0.7, beta_diversity=0.3)
        self.updater = LoRAUpdater(base_model=base_model, output_dir=lora_output_dir)

    async def run_pipeline3_for_scenarios(self, scenarios: list[ProjectScenario]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for s in scenarios:
            logger.info(
                "Running pipeline3 for user=%s db=%s desc=%s",
                s.user_id,
                s.neo4j_database,
                s.description,
            )
            out = await run_pipeline3(s.user_id, neo4j_database=s.neo4j_database)
            outputs.append({"scenario": asdict(s), "pipeline3_result": out})
        return outputs

    def upload_all_user_docs_to_rag(self, user_id: str, source_dir: Path) -> dict[str, Any]:
        return self.rag_uploader.upload_user_docs(user_id=user_id, source_dir=source_dir)

    def train_from_rollouts(
        self,
        rollouts: list[CursorRollout],
        *,
        epochs: int = 1,
    ) -> dict[str, Any]:
        grouped: dict[str, list[CursorRollout]] = {}
        for r in rollouts:
            grouped.setdefault(r.prompt, []).append(r)

        reward_rows: list[RewardRecord] = []
        for prompt, group in grouped.items():
            if len(group) < 2:
                logger.warning("Prompt group has <2 rollouts; GRPO will be weak: %s", prompt[:80])
            reward_rows.extend(self.rewarder.score_group(group))

        fit_result = self.updater.fit(rollouts, reward_rows, epochs=epochs)
        return {
            "fit_result": fit_result,
            "reward_count": len(reward_rows),
            "reward_preview": [asdict(r) for r in reward_rows[:10]],
        }


def _default_scenarios() -> list[ProjectScenario]:
    # Includes varying complexity and one "combined projects" style user.
    return [
        ProjectScenario(
            user_id="u_simple",
            neo4j_database="kg_simple",
            description="single medium project",
            project_roots=["project_a"],
        ),
        ProjectScenario(
            user_id="u_complex",
            neo4j_database="kg_complex",
            description="single complex multi-module project",
            project_roots=["project_big"],
        ),
        ProjectScenario(
            user_id="u_combo",
            neo4j_database="kg_combo",
            description="multiple projects merged for one user",
            project_roots=["project_big", "project_ml", "project_api"],
        ),
    ]


def _load_rollouts(path: Path) -> list[CursorRollout]:
    rows = _jsonl_read(path)
    out: list[CursorRollout] = []
    for r in rows:
        out.append(
            CursorRollout(
                user_id=str(r["user_id"]),
                neo4j_database=str(r["neo4j_database"]),
                rag_corpus=str(r["rag_corpus"]),
                prompt=str(r["prompt"]),
                agent_id=str(r["agent_id"]),
                changed_code=str(r["changed_code"]),
                metadata=dict(r.get("metadata") or {}),
            )
        )
    return out


def _save_reward_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = KGRAGRlAgent(
        rag_corpus_resource=args.rag_corpus,
        base_model=args.base_model,
        lora_output_dir=Path(args.lora_output_dir),
    )

    scenarios = _default_scenarios()
    pipeline_results = await agent.run_pipeline3_for_scenarios(scenarios)

    rag_uploads: list[dict[str, Any]] = []
    for s in scenarios:
        # Convention: local source roots are passed as --user-source-root/<user_id>
        user_source = Path(args.user_source_root) / s.user_id
        rag_uploads.append(agent.upload_all_user_docs_to_rag(user_id=s.user_id, source_dir=user_source))

    rollouts = _load_rollouts(Path(args.rollouts_jsonl))
    train_result = agent.train_from_rollouts(rollouts, epochs=args.epochs)

    full_report = {
        "pipeline3": pipeline_results,
        "rag_uploads": rag_uploads,
        "train": train_result,
    }
    _save_reward_report(Path(args.report_path), full_report)
    logger.info("RL training report written to %s", args.report_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "KG + RAG RL agent trainer: run pipeline3 scenarios, upload docs to RAG, "
            "score 10 Cursor outputs with GRPO-style reward, and LoRA-update a base model."
        )
    )
    p.add_argument(
        "--rag-corpus",
        required=True,
        help="Vertex RAG corpus resource, e.g. projects/.../locations/.../ragCorpora/...",
    )
    p.add_argument(
        "--rollouts-jsonl",
        required=True,
        help="JSONL file containing cursor rollout records (10 agents recommended).",
    )
    p.add_argument(
        "--user-source-root",
        required=True,
        help="Root folder with per-user Python source dirs, e.g. ./synthetic_users/<user_id>/...",
    )
    p.add_argument("--base-model", default=os.environ.get("RL_BASE_MODEL", "gpt2"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-output-dir", default="./artifacts/lora_adapter")
    p.add_argument("--report-path", default="./artifacts/rl_training_report.json")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
