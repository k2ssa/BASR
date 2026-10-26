from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selective LLM + ANEMONE experiments")
    parser.add_argument("--config", default=None, help="YAML configuration path")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--budget", type=float, default=None, help="Router node budget ratio")
    parser.add_argument(
        "--router",
        choices=[
            "fixed_top_fraction",
            "dynamic_mad",
            "self_calibrated",
            "adaptive_internal",
            "random",
            "whole_document",
        ],
        default=None,
    )
    parser.add_argument(
        "--router-seed",
        type=int,
        default=None,
        help="Random seed used by the random routing baseline",
    )
    parser.add_argument(
        "--llm-mode", choices=["cache", "routed_cache", "live"], default=None
    )
    parser.add_argument(
        "--attribute-fusion",
        choices=["weighted", "probability_or", "adaptive_evidence"],
        default=None,
        help="Attribute evidence fusion: fixed weights, rank Probability-OR, or node-adaptive evidence weights",
    )
    parser.add_argument(
        "--semantic-change-weight",
        type=float,
        default=None,
        help="Weight of the LLM semantic-change score in weighted attribute fusion",
    )
    parser.add_argument(
        "--neighbor-gain-weight",
        type=float,
        default=None,
        help="Weight of the neighborhood-consistency gain in weighted attribute fusion",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Re-run live LLM inference even when matching JSONL records exist",
    )
    parser.add_argument(
        "--force-node-embeddings",
        action="store_true",
        help="Re-encode every original node text with the configured BGE model",
    )
    parser.add_argument("--force-sentence-embeddings", action="store_true")
    parser.add_argument(
        "--force-repair-embeddings",
        action="store_true",
        help="Re-encode LLM-repaired texts even when a matching BGE cache exists",
    )
    parser.add_argument(
        "--train-anemone",
        "--train-graph",
        dest="train_anemone",
        action="store_true",
        help="Train the graph branch instead of loading an existing score CSV",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.dataset:
        config.dataset = args.dataset
    if args.project_root:
        config.project_root = args.project_root
    if args.budget is not None:
        config.router.budget_ratio = args.budget
    if args.router:
        config.router.method = args.router
    if args.router_seed is not None:
        config.router.seed = args.router_seed
    if args.llm_mode:
        config.llm.mode = args.llm_mode
    if args.attribute_fusion:
        config.attribute.fusion = args.attribute_fusion
    if args.semantic_change_weight is not None:
        config.attribute.semantic_change_weight = args.semantic_change_weight
    if args.neighbor_gain_weight is not None:
        config.attribute.neighbor_gain_weight = args.neighbor_gain_weight
    config.validate()
    result = run_experiment(
        config,
        force_node_embeddings=args.force_node_embeddings,
        force_sentence_embeddings=args.force_sentence_embeddings,
        force_repair_embeddings=args.force_repair_embeddings,
        force_llm=args.force_llm,
        train_anemone=args.train_anemone,
    )
    print(f"\n[output] {result.selection_path}")
    print(f"[output] {result.scores_path}")
    print(f"[output] {result.metrics_path}")
