from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EmbeddingConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "/models/bge-m3-en-v1.5"
    tokenizer: str = "model/bge-m3-en-v1.5"
    batch_size: int = 8
    max_tokens: int = 511


@dataclass
class RouterConfig:
    method: str = "fixed_top_fraction"
    budget_ratio: float = 0.30
    seed: int = 1
    sentence_fraction: float = 0.20
    min_sentence_chars: int = 12
    within_weight: float = 0.65
    intra_weight: float = 0.70


@dataclass
class LLMConfig:
    mode: str = "cache"
    version: str = "v9"
    output_tag: str | None = None
    provider: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-v4-pro"
    thinking: bool = False
    json_mode: bool = False
    retry_errors_on_resume: bool = True
    concurrency: int = 8
    max_neighbors: int = 5
    neighbor_text_chars: int = 300
    max_tokens: int = 512
    max_tokens_schedule: list[int] | None = None
    retries: int = 3

    @property
    def artifact_tag(self) -> str:
        return self.output_tag or self.version


@dataclass
class AttributeConfig:
    fusion: str = "weighted"
    router_weight: float = 0.0
    semantic_change_weight: float = 0.9
    neighbor_gain_weight: float = 0.1


@dataclass
class AnemoneConfig:
    score_file: str | None = None
    score_column: str = "anemone_score"
    seed: int = 1
    device: str = "auto"
    epochs: int = 100
    patience: int = 100
    batch_size: int = 300
    embedding_dim: int = 64
    subgraph_size: int = 4
    eval_rounds: int = 256
    alpha: float = 0.8
    lr: float = 0.001
    weight_decay: float = 0.0
    restart_prob: float = 0.9
    feature_normalization: str = "official-row"
    readout: str = "avg"
    negative_patch: int = 1
    negative_context: int = 1


@dataclass
class ExperimentConfig:
    project_root: str = "."
    dataset: str = "Cora"
    data_root: str = "data/processed"
    output_root: str = "outputs/llm_anemone"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    attribute: AttributeConfig = field(default_factory=AttributeConfig)
    anemone: AnemoneConfig = field(default_factory=AnemoneConfig)

    @property
    def root(self) -> Path:
        return Path(self.project_root).resolve()

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.data_root / self.dataset

    @property
    def result_dir(self) -> Path:
        return self.root / self.output_root / self.dataset

    @property
    def prompt_dir(self) -> Path:
        return Path(__file__).parent / "prompts"

    def validate(self) -> None:
        if not 0 < self.router.budget_ratio <= 1:
            raise ValueError("router.budget_ratio must be in (0, 1]")
        valid_router_methods = {
            "fixed_top_fraction",
            "dynamic_mad",
            "self_calibrated",
            "adaptive_internal",
            "random",
            "whole_document",
        }
        if self.router.method not in valid_router_methods:
            raise ValueError(
                "router.method must be fixed_top_fraction, dynamic_mad, "
                "self_calibrated, adaptive_internal, random, or whole_document"
            )
        if self.router.seed < 0:
            raise ValueError("router.seed must be nonnegative")
        if not 0 < self.router.sentence_fraction <= 1:
            raise ValueError("router.sentence_fraction must be in (0, 1]")
        valid_attribute_fusions = {"weighted", "probability_or", "adaptive_evidence"}
        if self.attribute.fusion not in valid_attribute_fusions:
            raise ValueError(
                "attribute.fusion must be weighted, probability_or, or adaptive_evidence"
            )
        weights = [
            self.attribute.router_weight,
            self.attribute.semantic_change_weight,
            self.attribute.neighbor_gain_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("attribute weights must be nonnegative")
        if self.attribute.fusion == "weighted" and sum(weights) <= 0:
            raise ValueError("attribute weights must sum to > 0 for weighted fusion")
        if self.llm.mode not in {"cache", "routed_cache", "live"}:
            raise ValueError("llm.mode must be cache, routed_cache, or live")
        if self.llm.provider not in {"dashscope", "deepseek", "openai_compatible"}:
            raise ValueError(
                "llm.provider must be dashscope, deepseek, or openai_compatible"
            )
        if self.embedding.batch_size < 1 or self.llm.concurrency < 1:
            raise ValueError("embedding.batch_size and llm.concurrency must be positive")
        if self.llm.max_tokens < 1:
            raise ValueError("llm.max_tokens must be positive")
        if self.llm.max_tokens_schedule is not None:
            if not self.llm.max_tokens_schedule or any(
                value < 1 for value in self.llm.max_tokens_schedule
            ):
                raise ValueError(
                    "llm.max_tokens_schedule must contain positive integers"
                )
        if self.llm.retries < 1:
            raise ValueError("llm.retries must be positive")
        if self.anemone.epochs < 1 or self.anemone.eval_rounds < 1:
            raise ValueError("anemone.epochs and anemone.eval_rounds must be positive")
        if not 0 <= self.anemone.alpha <= 1:
            raise ValueError("anemone.alpha must be in [0, 1]")


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        _merge_dataclass(cfg, values)
    cfg.project_root = os.getenv("PROJECT_ROOT", cfg.project_root)
    cfg.dataset = os.getenv("DATASET", cfg.dataset)
    cfg.embedding.base_url = os.getenv("EMBED_BASE_URL", cfg.embedding.base_url)
    cfg.embedding.api_key = os.getenv("EMBED_API_KEY", cfg.embedding.api_key)
    cfg.embedding.model = os.getenv("EMBED_MODEL", cfg.embedding.model)
    cfg.llm.base_url = os.getenv("LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.model = os.getenv("LLM_MODEL", cfg.llm.model)
    cfg.validate()
    return cfg
