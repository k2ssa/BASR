from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import scipy.sparse as sp
import yaml


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    template: str
    needs_neighbors: bool


def load_prompt(prompt_dir: Path, version: str) -> PromptTemplate:
    path = prompt_dir / f"teacher_{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version {version!r} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    template = value["template"]
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    return PromptTemplate(
        system=value.get("system", "Return only valid JSON when asked."),
        template=template,
        needs_neighbors="neighbor_texts" in placeholders,
    )


def render_prompt(
    prompt: PromptTemplate,
    node_index: int,
    node_id: str,
    texts: list[str],
    adjacency: sp.csr_matrix,
    max_neighbors: int,
    neighbor_chars: int,
) -> str:
    values = {"node_id": node_id, "node_text": texts[node_index]}
    if prompt.needs_neighbors:
        indices = adjacency.getrow(node_index).indices.tolist()[:max_neighbors]
        values.update(
            neighbor_texts=(
                "\n\n".join(
                    f"[Neighbor {row + 1}] {texts[index][:neighbor_chars]}"
                    for row, index in enumerate(indices)
                )
                if indices
                else "(no neighbors)"
            ),
            num_neighbors=len(indices),
        )
    return prompt.template.format(**values).strip()
