from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm

from .config import EmbeddingConfig


class BGEEncoder:
    """OpenAI-compatible client for the project's local BGE service."""

    def __init__(self, config: EmbeddingConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._tokenizer = None

    @property
    def signature(self) -> str:
        value = f"{self.config.base_url}|{self.config.model}|{self.config.tokenizer}|{self.config.max_tokens}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = Path(self.config.tokenizer)
            if not path.is_absolute():
                path = self.project_root / path
            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        return self._tokenizer

    def _truncate(self, texts: list[str]) -> list[str]:
        tokenizer = self._get_tokenizer()
        result = []
        for text in texts:
            ids = tokenizer.encode(
                str(text), add_special_tokens=True, verbose=False
            )
            if len(ids) > self.config.max_tokens:
                ids = ids[: self.config.max_tokens]
                text = tokenizer.decode(ids, skip_special_tokens=True)
            result.append(str(text))
        return result

    def encode(self, texts: list[str], description: str = "BGE encoding") -> torch.Tensor:
        texts = self._truncate(texts)
        vectors: list[list[float]] = []
        for start in tqdm(
            range(0, len(texts), self.config.batch_size), description
        ):
            response = self.client.embeddings.create(
                model=self.config.model,
                input=texts[start : start + self.config.batch_size],
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return torch.tensor(np.asarray(vectors), dtype=torch.float32)


def text_digest(groups: list[list[str]]) -> str:
    raw = json.dumps(groups, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_node_embeddings(dataset_dir: Path, dataset: str, n: int) -> torch.Tensor:
    path = dataset_dir / f"{dataset}_exp_x0_emb.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing freshly generated node embedding cache: {path}. "
            "Run `python -m llm_anemone.embed_nodes --config <config> --force` first. "
            "Legacy BAAI embeddings.npy files are intentionally not used."
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get("embedding", value.get("embeddings"))
    tensor = torch.as_tensor(value).float()
    if tensor.shape[0] != n:
        raise ValueError(
            f"Node embedding row mismatch in {path}: expected {n}, got {tensor.shape[0]}"
        )
    print(f"[BGE] node cache: {path} {tuple(tensor.shape)}")
    return tensor


def load_or_encode_node_embeddings(
    dataset_dir: Path,
    dataset: str,
    texts: list[str],
    encoder: BGEEncoder,
    force: bool = False,
) -> torch.Tensor:
    """Load the canonical node cache or encode every original node text.

    Legacy ``BAAI/.../embeddings.npy`` files are deliberately ignored so that
    all branches use embeddings produced by the configured encoder.
    """
    path = dataset_dir / f"{dataset}_exp_x0_emb.pt"
    if path.exists() and not force:
        return load_node_embeddings(dataset_dir, dataset, len(texts))

    embeddings = encoder.encode(
        texts,
        description=f"Encoding {dataset} node texts",
    ).cpu().float()
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Node embedding row mismatch: expected {len(texts)}, "
            f"got {embeddings.shape[0]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)
    print(f"[BGE] saved node cache: {path} {tuple(embeddings.shape)}")
    return embeddings


def load_or_encode_sentences(
    cache_path: Path,
    sentence_groups: list[list[str]],
    encoder: BGEEncoder,
    force: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    digest = text_digest(sentence_groups)
    if cache_path.exists() and not force:
        value = torch.load(cache_path, map_location="cpu", weights_only=True)
        cache_signature = value.get("encoder_signature")
        # Old MyGAD sentence caches predate signatures. They remain compatible
        # with the original experiment; newly generated caches are strict.
        if value.get("digest") == digest and (
            cache_signature is None or cache_signature == encoder.signature
        ):
            embedding = value["embedding"].float()
            offsets = torch.as_tensor(value["offsets"]).numpy().astype(np.int64)
            print(f"[BGE] sentence cache: {cache_path} {tuple(embedding.shape)}")
            return embedding, offsets

    flat = [sentence for group in sentence_groups for sentence in group]
    offsets = np.zeros(len(sentence_groups) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(group) for group in sentence_groups])
    embedding = encoder.encode(flat, "Encoding node sentences").cpu().float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "digest": digest,
            "encoder_signature": encoder.signature,
            "embedding": embedding,
            "offsets": torch.from_numpy(offsets),
        },
        cache_path,
    )
    print(f"[BGE] saved sentence cache: {cache_path}")
    return embedding, offsets


def load_clean_embedding_cache(
    dataset_dir: Path, dataset: str, version: str, repair_nodes: np.ndarray
) -> torch.Tensor:
    path = dataset_dir / f"{dataset}_all_{version}_clean_emb.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing clean embedding cache: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    cached_nodes = torch.as_tensor(value["node_idx"]).numpy().astype(np.int64)
    if not np.array_equal(cached_nodes, repair_nodes):
        raise ValueError("LLM JSONL repair order differs from clean embedding cache")
    return torch.as_tensor(value["embedding"]).float()
