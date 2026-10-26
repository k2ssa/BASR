from __future__ import annotations

import json
import os
import re
import threading
import hashlib
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from ..config import LLMConfig
from ..data import GraphData
from .cache import LLMRepairSet, load_repairs
from .prompting import load_prompt, render_prompt


def extract_json_object(raw: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON object in LLM response: {raw[:200]!r}")


def _provider_extra_body(config: LLMConfig) -> dict:
    if config.provider == "deepseek":
        return {
            "thinking": {
                "type": "enabled" if config.thinking else "disabled",
            }
        }
    if config.provider == "dashscope":
        return {"enable_thinking": config.thinking}
    return {}


def _api_key(config: LLMConfig) -> str:
    if config.provider == "deepseek":
        value = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        expected = "DEEPSEEK_API_KEY or LLM_API_KEY"
    else:
        value = os.getenv("LLM_API_KEY")
        expected = "LLM_API_KEY"
    if not value:
        raise EnvironmentError(f"Set {expected} before using llm.mode=live")
    return value


def _call(client: OpenAI, config: LLMConfig, system: str, user: str) -> tuple[dict, dict]:
    last_error: Exception | None = None
    token_schedule = config.max_tokens_schedule or [config.max_tokens]
    for attempt in range(1, config.retries + 1):
        attempt_max_tokens = token_schedule[min(attempt - 1, len(token_schedule) - 1)]
        try:
            request = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": attempt_max_tokens,
                "extra_body": _provider_extra_body(config),
            }
            if config.json_mode:
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(
                **request,
            )
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            metadata = {
                "raw_response": raw,
                "finish_reason": response.choices[0].finish_reason,
                "attempt": attempt,
                "max_tokens": attempt_max_tokens,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
            return extract_json_object(raw), metadata
        except Exception as error:
            last_error = error
    raise RuntimeError(f"LLM request failed after {config.retries} attempts: {last_error}")


def infer_candidates(
    graph: GraphData,
    candidates: np.ndarray,
    output_path: Path,
    prompt_dir: Path,
    config: LLMConfig,
    force: bool = False,
) -> LLMRepairSet:
    """Infer only routed nodes; JSONL writes are incremental and resumable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt(prompt_dir, config.version)
    signature_source = json.dumps(
        {
            "version": config.version,
            "provider": config.provider,
            "base_url": config.base_url,
            "model": config.model,
            "thinking": config.thinking,
            "json_mode": config.json_mode,
            "system": prompt.system,
            "template": prompt.template,
            "max_neighbors": config.max_neighbors,
            "neighbor_text_chars": config.neighbor_text_chars,
            "max_tokens": config.max_tokens,
            "max_tokens_schedule": config.max_tokens_schedule,
            "retries": config.retries,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    inference_signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
    inference_run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    completed = set()
    if output_path.exists() and not force:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("inference_signature") == inference_signature:
                        completed.add(int(record["node_idx"]))
        if not config.retry_errors_on_resume:
            errors = output_path.with_name(output_path.stem + "_errors.jsonl")
            if errors.exists():
                with errors.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("inference_signature") == inference_signature:
                                completed.add(int(record["node_idx"]))
    pending = [int(node) for node in candidates if int(node) not in completed]
    if not pending:
        return load_repairs(
            output_path,
            graph.num_nodes,
            inference_signature=inference_signature,
        ).selected(candidates)
    api_key = _api_key(config)
    client = OpenAI(base_url=config.base_url, api_key=api_key)
    lock = threading.Lock()

    def work(node: int) -> dict:
        user = render_prompt(
            prompt, node, graph.node_ids[node], graph.texts, graph.adjacency,
            config.max_neighbors, config.neighbor_text_chars,
        )
        result, response_metadata = _call(client, config, prompt.system, user)
        return {
            "node_idx": node,
            "node_id": graph.node_ids[node],
            "prompt_ver": config.version,
            "llm_model": config.model,
            "inference_signature": inference_signature,
            "inference_run_id": inference_run_id,
            "inferred_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_text": graph.texts[node],
            "llm_result": result,
            **response_metadata,
        }

    errors = output_path.with_name(output_path.stem + "_errors.jsonl")
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {pool.submit(work, node): node for node in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM inference"):
            node = futures[future]
            try:
                record = future.result()
                target = output_path
            except Exception as error:
                record = {
                    "node_idx": node,
                    "node_id": graph.node_ids[node],
                    "prompt_ver": config.version,
                    "llm_model": config.model,
                    "inference_signature": inference_signature,
                    "inference_run_id": inference_run_id,
                    "inferred_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                }
                target = errors
            with lock, target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return load_repairs(
        output_path,
        graph.num_nodes,
        inference_signature=inference_signature,
    ).selected(candidates)
