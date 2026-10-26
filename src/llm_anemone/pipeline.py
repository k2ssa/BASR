from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .anemone import train_and_score
from .config import ExperimentConfig
from .data import load_graph, load_labels
from .embeddings import (
    BGEEncoder,
    load_clean_embedding_cache,
    load_or_encode_node_embeddings,
    load_or_encode_sentences,
)
from .evaluation import evaluate
from .llm import infer_candidates, load_repairs
from .routing import neighbor_means, route, split_sentences
from .scoring import compute_attribute_scores, fuse_scores


@dataclass(frozen=True)
class ExperimentResult:
    metrics: pd.DataFrame
    scores_path: Path
    metrics_path: Path
    selection_path: Path


def _find_anemone_scores(config: ExperimentConfig) -> Path | None:
    if config.anemone.score_file:
        path = Path(config.anemone.score_file)
        return path if path.is_absolute() else config.root / path
    preferred = config.dataset_dir / (
        f"{config.dataset}_official_anemone_seed1_a0p8_official_row_scores.csv"
    )
    if preferred.exists():
        return preferred
    candidates = sorted(config.dataset_dir.glob(f"{config.dataset}_official_anemone*_scores.csv"))
    return candidates[0] if candidates else None


def _load_or_train_anemone(
    config: ExperimentConfig,
    features: torch.Tensor,
    force_train: bool,
) -> tuple[np.ndarray, Path]:
    path = _find_anemone_scores(config)
    if path and path.exists() and not force_train:
        frame = pd.read_csv(path)
        if config.anemone.score_column not in frame:
            raise KeyError(f"{config.anemone.score_column!r} not found in {path}")
        values = frame[config.anemone.score_column].to_numpy(dtype=np.float32)
        return values, path
    if not force_train:
        raise FileNotFoundError("No ANEMONE scores found; run the anemone command or add --train-anemone")
    config.result_dir.mkdir(parents=True, exist_ok=True)
    training_config = asdict(config.anemone)
    training_config["score_file"] = None
    training_fingerprint = hashlib.sha256(
        json.dumps(training_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    training_tag = (
        f"anemone_seed{config.anemone.seed}_a{config.anemone.alpha:g}_"
        f"{config.anemone.feature_normalization}_{training_fingerprint}"
    ).replace(".", "p")
    checkpoint = config.result_dir / f"{training_tag}.pt"
    graph = load_graph(config.dataset_dir, config.dataset)
    scores = train_and_score(features, graph.adjacency, config.anemone, checkpoint)
    path = config.result_dir / f"{training_tag}_scores.csv"
    pd.DataFrame({"node_idx": np.arange(graph.num_nodes), **scores}).to_csv(path, index=False)
    return scores[config.anemone.score_column], path


def _repair_embeddings(
    config: ExperimentConfig,
    graph_size: int,
    feature_dim: int,
    candidates: np.ndarray,
    encoder: BGEEncoder,
    force_llm: bool,
    force_embeddings: bool,
) -> tuple[np.ndarray, torch.Tensor, int]:
    artifact_tag = config.llm.artifact_tag
    if config.llm.mode == "cache":
        jsonl = config.dataset_dir / f"{config.dataset}_full_infer_{artifact_tag}.jsonl"
        all_repairs = load_repairs(jsonl, graph_size)
        all_embeddings = load_clean_embedding_cache(
            config.dataset_dir, config.dataset, artifact_tag, all_repairs.nodes
        )
        row_by_node = {int(node): row for row, node in enumerate(all_repairs.nodes)}
        nodes = np.asarray([node for node in candidates if int(node) in row_by_node], dtype=np.int64)
        rows = [row_by_node[int(node)] for node in nodes]
        return nodes, all_embeddings[rows], all_repairs.record_count

    if config.llm.mode == "routed_cache":
        jsonl = config.result_dir / f"{config.dataset}_routed_{artifact_tag}.jsonl"
        prompts: set[str] = set()
        models: set[str] = set()
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    prompts.add(str(record.get("prompt_ver", "")))
                    models.add(str(record.get("llm_model", "")))
        if prompts != {config.llm.version} or models != {config.llm.model}:
            raise ValueError(
                "Routed LLM cache metadata differs from the configured prompt/model: "
                f"prompts={sorted(prompts)}, models={sorted(models)}"
            )
        all_repairs = load_repairs(jsonl, graph_size)
        cache = config.result_dir / f"{config.dataset}_routed_{artifact_tag}_clean_emb.pt"
        value = torch.load(cache, map_location="cpu", weights_only=True)
        cached_nodes = torch.as_tensor(value["node_idx"]).numpy().astype(np.int64)
        if not np.array_equal(cached_nodes, all_repairs.nodes):
            raise ValueError("Routed JSONL and clean embedding cache are misaligned")
        cached_embeddings = torch.as_tensor(value["embedding"]).float()
        row_by_node = {int(node): row for row, node in enumerate(cached_nodes)}
        nodes = np.asarray(
            [node for node in candidates if int(node) in row_by_node], dtype=np.int64
        )
        rows = [row_by_node[int(node)] for node in nodes]
        print(
            f"[BGE] selecting {len(rows)} repairs from the frozen "
            f"{all_repairs.record_count}-response cache "
            f"({len(cached_nodes)} corrections available)"
        )
        return nodes, cached_embeddings[rows], all_repairs.record_count

    graph = load_graph(config.dataset_dir, config.dataset)
    jsonl = config.result_dir / f"{config.dataset}_routed_{artifact_tag}.jsonl"
    repairs = infer_candidates(
        graph, candidates, jsonl, config.prompt_dir, config.llm, force=force_llm
    )
    if len(repairs.nodes) == 0:
        return repairs.nodes, torch.empty((0, feature_dim), dtype=torch.float32), repairs.record_count
    cache = config.result_dir / f"{config.dataset}_routed_{artifact_tag}_clean_emb.pt"
    clean_digest = hashlib.sha256(
        json.dumps(repairs.clean_texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if cache.exists() and not force_embeddings:
        value = torch.load(cache, map_location="cpu", weights_only=True)
        cached_nodes = torch.as_tensor(value["node_idx"]).numpy().astype(np.int64)
        cached_embeddings = torch.as_tensor(value["embedding"]).float()
        if (
            np.array_equal(cached_nodes, repairs.nodes)
            and value.get("text_digest") == clean_digest
            and value.get("encoder_signature") == encoder.signature
        ):
            return repairs.nodes, cached_embeddings, repairs.record_count
        if value.get("encoder_signature") == encoder.signature:
            text_by_node = {
                int(node): text
                for node, text in zip(repairs.nodes, repairs.clean_texts)
            }
            cached_node_set = set(map(int, cached_nodes))
            repair_node_set = set(map(int, repairs.nodes))
            if repair_node_set.issubset(cached_node_set):
                row_by_node = {
                    int(node): row for row, node in enumerate(cached_nodes)
                }
                rows = [row_by_node[int(node)] for node in repairs.nodes]
                print(
                    f"[BGE] selecting {len(rows)} repairs from the "
                    f"{len(cached_nodes)}-node superset cache"
                )
                return repairs.nodes, cached_embeddings[rows], repairs.record_count
            if cached_node_set.issubset(text_by_node):
                cached_texts = [text_by_node[int(node)] for node in cached_nodes]
                cached_digest = hashlib.sha256(
                    json.dumps(
                        cached_texts,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if value.get("text_digest") == cached_digest:
                    missing_nodes = np.asarray(
                        [
                            int(node)
                            for node in repairs.nodes
                            if int(node) not in cached_node_set
                        ],
                        dtype=np.int64,
                    )
                    if len(missing_nodes):
                        missing_texts = [text_by_node[int(node)] for node in missing_nodes]
                        print(
                            f"[BGE] incrementally encoding {len(missing_nodes)} "
                            "new LLM repairs"
                        )
                        missing_embeddings = encoder.encode(
                            missing_texts,
                            "Encoding new LLM repairs",
                        )
                        by_node = {
                            int(node): cached_embeddings[row]
                            for row, node in enumerate(cached_nodes)
                        }
                        by_node.update(
                            {
                                int(node): missing_embeddings[row]
                                for row, node in enumerate(missing_nodes)
                            }
                        )
                        embeddings = torch.stack(
                            [by_node[int(node)] for node in repairs.nodes]
                        )
                        torch.save(
                            {
                                "node_idx": torch.from_numpy(repairs.nodes),
                                "text_digest": clean_digest,
                                "encoder_signature": encoder.signature,
                                "embedding": embeddings,
                            },
                            cache,
                        )
                        return repairs.nodes, embeddings, repairs.record_count
    embeddings = encoder.encode(repairs.clean_texts, "Encoding LLM repairs")
    torch.save(
        {
            "node_idx": torch.from_numpy(repairs.nodes),
            "text_digest": clean_digest,
            "encoder_signature": encoder.signature,
            "embedding": embeddings,
        },
        cache,
    )
    return repairs.nodes, embeddings, repairs.record_count


def run_experiment(
    config: ExperimentConfig,
    force_node_embeddings: bool = False,
    force_sentence_embeddings: bool = False,
    force_repair_embeddings: bool = False,
    force_llm: bool = False,
    train_anemone: bool = False,
) -> ExperimentResult:
    config.validate()
    config.result_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(config.dataset_dir, config.dataset)
    encoder = BGEEncoder(config.embedding, config.root)
    original = load_or_encode_node_embeddings(
        config.dataset_dir,
        config.dataset,
        graph.texts,
        encoder,
        force=force_node_embeddings,
    )
    neighbors = neighbor_means(original, graph.adjacency)

    groups = [split_sentences(text, config.router.min_sentence_chars) for text in graph.texts]
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum([len(group) for group in groups])]
    )
    sentence_embeddings: torch.Tensor | None = None
    if config.router.method not in {"random", "whole_document"}:
        sentence_cache = config.result_dir / f"{config.dataset}_sentence_embeddings.pt"
        legacy_sentence_cache = config.dataset_dir / f"{config.dataset}_sentence_router_embeddings.pt"
        if not sentence_cache.exists() and legacy_sentence_cache.exists() and not force_sentence_embeddings:
            sentence_cache = legacy_sentence_cache
        sentence_embeddings, offsets = load_or_encode_sentences(
            sentence_cache, groups, encoder, force_sentence_embeddings
        )
    has_neighbors = np.asarray(graph.adjacency.sum(axis=1)).ravel() > 0
    routed = route(
        sentence_embeddings,
        offsets,
        neighbors,
        config.router,
        has_neighbors=has_neighbors,
        node_embeddings=original,
    )
    repair_nodes, repair_embeddings, record_count = _repair_embeddings(
        config, graph.num_nodes, original.shape[1], routed.candidate_indices, encoder,
        force_llm, force_repair_embeddings,
    )
    if repair_embeddings.ndim != 2 or (
        len(repair_nodes) and repair_embeddings.shape[1] != original.shape[1]
    ):
        raise ValueError(
            "Clean-text BGE embedding dimension does not match original node embeddings"
        )
    attribute = compute_attribute_scores(
        original, repair_nodes, repair_embeddings, neighbors, routed.score, config.attribute
    )
    anemone_raw, anemone_path = _load_or_train_anemone(
        config, original, train_anemone
    )
    if len(anemone_raw) != graph.num_nodes:
        raise ValueError("ANEMONE score count does not match graph size")
    fused = fuse_scores(attribute.attribute, anemone_raw)

    scores = {
        "router_score": routed.score,
        "router_intra": routed.intra,
        "router_neighbor": routed.neighbor,
        "router_max_pair": routed.max_pair,
        "llm_semantic_change": attribute.semantic_change,
        "llm_neighbor_gain": attribute.neighbor_gain,
        "attr_score": attribute.attribute,
        "anemone_raw": anemone_raw,
        **fused,
    }
    # Detector construction ends here. Labels enter only for evaluation/export.
    labels = load_labels(config.dataset_dir, config.dataset)
    metrics = evaluate(scores, labels)
    # Routing weights are diagnostics rather than anomaly signals, so append them
    # only after evaluation to avoid meaningless AUC/AP rows.
    if routed.center_weight is not None:
        scores["router_center_weight"] = routed.center_weight
    if routed.pair_weight is not None:
        scores["router_pair_weight"] = routed.pair_weight
    candidate_mask = routed.candidate_mask
    repaired_mask = np.zeros(graph.num_nodes, dtype=bool)
    repaired_mask[repair_nodes] = True

    attr = config.attribute
    fingerprint_source = {
        "dataset": config.dataset,
        "router": asdict(config.router),
        "llm_version": config.llm.version,
        "llm_output_tag": config.llm.artifact_tag,
        "llm_mode": config.llm.mode,
        "attribute": asdict(config.attribute),
        "anemone_column": config.anemone.score_column,
        "anemone_score_file": str(anemone_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    tag = (
        f"{config.router.method}_b{int(round(config.router.budget_ratio * 100)):02d}_"
        f"{config.llm.artifact_tag}_af{attr.fusion}"
    )
    if attr.fusion == "weighted":
        tag += (
            f"_rw{attr.router_weight:g}_cw{attr.semantic_change_weight:g}_"
            f"gw{attr.neighbor_gain_weight:g}"
        )
    tag = f"{tag}_{fingerprint}".replace(".", "p")
    selection_path = config.result_dir / f"{tag}_selection.csv"
    scores_path = config.result_dir / f"{tag}_scores.csv"
    metrics_path = config.result_dir / f"{tag}_metrics.csv"
    pd.DataFrame(
        {
            "node_idx": np.arange(graph.num_nodes),
            "node_id": graph.node_ids,
            "sentence_count": np.diff(offsets),
            "candidate": candidate_mask,
            "llm_repair_applied": repaired_mask,
            "attr_anomaly_label": labels.attribute.astype(int),
            "stru_anomaly_label": labels.structure.astype(int),
            "unified_anomaly_label": labels.unified.astype(int),
        }
    ).to_csv(selection_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"node_idx": np.arange(graph.num_nodes), **scores}).to_csv(
        scores_path, index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(
        f"[route] candidates={candidate_mask.sum()}/{graph.num_nodes}, "
        f"LLM records={record_count}, repairs applied={repaired_mask.sum()}"
    )
    diagnostic_name = "attribute"
    diagnostic_labels = labels.attribute
    if np.unique(diagnostic_labels).size < 2:
        diagnostic_name = "unified"
        diagnostic_labels = labels.unified
    diagnostic_total = int(diagnostic_labels.sum())
    diagnostic_selected = int((candidate_mask & diagnostic_labels).sum())
    diagnostic_repaired = int((repaired_mask & diagnostic_labels).sum())
    diagnostic_selected_rate = (
        diagnostic_selected / diagnostic_total if diagnostic_total else float("nan")
    )
    diagnostic_repaired_rate = (
        diagnostic_repaired / diagnostic_total if diagnostic_total else float("nan")
    )
    print(
        f"[diagnostic labels only] {diagnostic_name} anomalies: "
        f"selected={diagnostic_selected}/{diagnostic_total} "
        f"({diagnostic_selected_rate:.1%}), "
        f"repaired={diagnostic_repaired}/{diagnostic_total} "
        f"({diagnostic_repaired_rate:.1%})"
    )
    print(f"[ANEMONE] {anemone_path} column={config.anemone.score_column}")
    focus = metrics[metrics.signal.isin(["router_score", "attr_score", "anemone_raw", "unified_max", "unified_prob_or"])]
    print("\n=== Selective LLM + official ANEMONE ===")
    print(focus.to_string(index=False))
    return ExperimentResult(metrics, scores_path, metrics_path, selection_path)
