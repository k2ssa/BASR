from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LLMRepairSet:
    nodes: np.ndarray
    clean_texts: list[str]
    record_count: int

    def selected(self, candidates: np.ndarray) -> "LLMRepairSet":
        allowed = set(map(int, candidates))
        keep = [row for row, node in enumerate(self.nodes) if int(node) in allowed]
        return LLMRepairSet(
            nodes=self.nodes[keep],
            clean_texts=[self.clean_texts[row] for row in keep],
            record_count=self.record_count,
        )


def _is_true(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def load_repairs(
    path: Path,
    num_nodes: int | None = None,
    inference_signature: str | None = None,
) -> LLMRepairSet:
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM output: {path}")
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if (
                    inference_signature is not None
                    and item.get("inference_signature") != inference_signature
                ):
                    continue
                records[int(item["node_idx"])] = item
    nodes, texts = [], []
    limit = num_nodes if num_nodes is not None else max(records, default=-1) + 1
    for node in range(limit):
        result = records.get(node, {}).get("llm_result", {})
        flag = result.get("need_correction", result.get("need_refine", False))
        clean = str(result.get("clean_text") or "").strip()
        if _is_true(flag) and clean:
            nodes.append(node)
            texts.append(clean)
    return LLMRepairSet(np.asarray(nodes, dtype=np.int64), texts, len(records))
