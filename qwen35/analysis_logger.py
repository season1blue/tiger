import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def normalize_binary_answer(text: str | None) -> str | None:
    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip().lower()
    if not cleaned:
        return None

    if any(token in cleaned.split() for token in ("no", "not")):
        return "no"
    if "yes" in cleaned.split():
        return "yes"

    if "no" in cleaned or "not" in cleaned:
        return "no"
    if "yes" in cleaned:
        return "yes"
    return None


def _safe_filename(value: str) -> str:
    # Keep alnum / dot / dash / underscore; replace everything else to avoid nested paths.
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", value)
    safe = safe.strip("._")
    return safe or "sample"


def _slice_prefix_inputs(inputs: dict[str, Any], prefix_len: int, input_key: str = "input_ids") -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    full_len = None
    input_ids = inputs.get(input_key)
    if isinstance(input_ids, torch.Tensor):
        full_len = input_ids.shape[-1]

    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            sliced[key] = value
            continue

        if key == input_key:
            sliced[key] = value[..., :prefix_len]
            continue

        if full_len is not None and value.ndim >= 2 and value.shape[-1] == full_len:
            sliced[key] = value[..., :prefix_len]
        else:
            sliced[key] = value

    return sliced


def _build_prefix_attention_mask(prefix_ids: torch.Tensor, original_mask: torch.Tensor | None) -> torch.Tensor | None:
    if original_mask is None:
        return torch.ones_like(prefix_ids)

    if original_mask.ndim == 1:
        original_mask = original_mask.unsqueeze(0)

    if original_mask.shape[-1] >= prefix_ids.shape[-1]:
        return original_mask[..., : prefix_ids.shape[-1]]

    pad_len = prefix_ids.shape[-1] - original_mask.shape[-1]
    pad = torch.ones(
        original_mask.shape[0],
        pad_len,
        dtype=original_mask.dtype,
        device=original_mask.device,
    )
    return torch.cat([original_mask, pad], dim=-1)


def _build_prefix_mm_token_type_ids(
    prefix_ids: torch.Tensor, original_token_types: torch.Tensor | None
) -> torch.Tensor | None:
    if original_token_types is None:
        return None

    if original_token_types.ndim == 1:
        original_token_types = original_token_types.unsqueeze(0)

    if original_token_types.shape[-1] >= prefix_ids.shape[-1]:
        return original_token_types[..., : prefix_ids.shape[-1]]

    pad_len = prefix_ids.shape[-1] - original_token_types.shape[-1]
    # Newly generated tokens are text tokens for Qwen2.5-VL, token type id = 0.
    pad = torch.zeros(
        original_token_types.shape[0],
        pad_len,
        dtype=original_token_types.dtype,
        device=original_token_types.device,
    )
    return torch.cat([original_token_types, pad], dim=-1)


def _mean_pool_image_states(hidden_states: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
    if image_mask.ndim == 2:
        image_mask = image_mask.unsqueeze(-1)

    mask = image_mask.to(dtype=hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (hidden_states * mask).sum(dim=1) / denom
    return pooled


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


@torch.inference_mode()
def trace_qwen25_sample(
    model,
    processor,
    inputs: dict[str, Any],
    generated_ids_trimmed: torch.Tensor,
    metadata: dict[str, Any],
    analysis_log_dir: str,
) -> Path:
    analysis_dir = Path(analysis_log_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    input_ids = inputs["input_ids"]
    if generated_ids_trimmed.ndim == 1:
        generated_ids_trimmed = generated_ids_trimmed.unsqueeze(0)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    prompt_len = input_ids.shape[-1]
    full_ids = torch.cat([input_ids, generated_ids_trimmed.to(input_ids.device)], dim=-1)
    image_token_id = int(model.config.image_token_id)
    hidden_layers = len(model.model.language_model.layers)
    layer_indices = list(range(hidden_layers + 1))

    pooled_states: list[torch.Tensor] = []
    direction_scores: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []
    entropy_scores: list[torch.Tensor] = []
    text_image_sims: list[torch.Tensor] = []
    step_records: list[dict[str, Any]] = []
    generated_tokens: list[str] = []

    tokenizer = getattr(processor, "tokenizer", None)

    for step_idx in range(generated_ids_trimmed.shape[-1]):
        prefix_len = prompt_len + step_idx + 1
        prefix_ids = full_ids[:, :prefix_len]
        prefix_inputs = _slice_prefix_inputs(inputs, prefix_len)
        prefix_inputs["input_ids"] = prefix_ids
        prefix_inputs["attention_mask"] = _build_prefix_attention_mask(
            prefix_ids, inputs.get("attention_mask")
        ).to(prefix_ids.device)
        mm_token_type_ids = _build_prefix_mm_token_type_ids(prefix_ids, inputs.get("mm_token_type_ids"))
        if mm_token_type_ids is not None:
            prefix_inputs["mm_token_type_ids"] = mm_token_type_ids.to(prefix_ids.device)

        outputs = model(
            **prefix_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen2.5-VL analysis tracing requires hidden_states=True")

        image_mask = prefix_ids.eq(image_token_id)
        if not torch.any(image_mask):
            raise RuntimeError("No image tokens were found while tracing the Qwen2.5-VL sample")

        step_layer_vectors: list[torch.Tensor] = []
        step_layer_entropy: list[torch.Tensor] = []
        step_layer_similarity: list[torch.Tensor] = []

        for layer_hidden in hidden_states:
            pooled = _mean_pool_image_states(layer_hidden, image_mask)
            pooled = pooled.squeeze(0)
            step_layer_vectors.append(pooled.detach().cpu())

            layer_logits = model.lm_head(layer_hidden[:, -1, :])
            step_layer_entropy.append(_entropy_from_logits(layer_logits).squeeze(0).detach().cpu())
            step_layer_similarity.append(
                F.cosine_similarity(layer_hidden[:, -1, :], pooled.unsqueeze(0), dim=-1).squeeze(0).detach().cpu()
            )

        step_layer_tensor = torch.stack(step_layer_vectors, dim=0)
        step_entropy_tensor = torch.stack(step_layer_entropy, dim=0)
        step_similarity_tensor = torch.stack(step_layer_similarity, dim=0)

        step_direction = torch.full((step_layer_tensor.shape[0],), float("nan"), dtype=step_layer_tensor.dtype)
        step_update_norm = torch.full((step_layer_tensor.shape[0],), float("nan"), dtype=step_layer_tensor.dtype)

        for layer_idx in range(1, step_layer_tensor.shape[0]):
            update = step_layer_tensor[layer_idx] - step_layer_tensor[layer_idx - 1]
            step_update_norm[layer_idx] = update.norm()

        for layer_idx in range(2, step_layer_tensor.shape[0]):
            prev_dir = step_layer_tensor[layer_idx - 1] - step_layer_tensor[layer_idx - 2]
            curr_dir = step_layer_tensor[layer_idx] - step_layer_tensor[layer_idx - 1]
            step_direction[layer_idx] = 1.0 - F.cosine_similarity(curr_dir, prev_dir, dim=0)

        pooled_states.append(step_layer_tensor)
        direction_scores.append(step_direction)
        update_norms.append(step_update_norm)
        entropy_scores.append(step_entropy_tensor)
        text_image_sims.append(step_similarity_tensor)

        token_id = int(generated_ids_trimmed[0, step_idx].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False) if tokenizer is not None else str(token_id)
        generated_tokens.append(token_text)

        step_records.append(
            {
                "step_index": step_idx + 1,
                "generated_token_id": token_id,
                "generated_token": token_text,
                "token_is_incorrect": None,
                "layer_indices": layer_indices,
                "direction_scores": step_direction.tolist(),
                "update_norms": step_update_norm.tolist(),
                "entropy_scores": step_entropy_tensor.tolist(),
                "text_image_similarity": step_similarity_tensor.tolist(),
            }
        )

    sample_id = str(metadata.get("sample_id", metadata.get("question_id", "sample")))
    sample_file_id = _safe_filename(sample_id)
    sample_path = analysis_dir / f"{sample_file_id}.pt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "metadata": metadata,
        "layer_indices": layer_indices,
        "generated_token_ids": generated_ids_trimmed.detach().cpu(),
        "generated_tokens": generated_tokens,
        "pooled_image_states": torch.stack(pooled_states, dim=0),
        "direction_scores": torch.stack(direction_scores, dim=0),
        "image_state_update_norms": torch.stack(update_norms, dim=0),
        "layer_entropy": torch.stack(entropy_scores, dim=0),
        "text_image_similarity": torch.stack(text_image_sims, dim=0),
        "steps": step_records,
    }
    torch.save(record, sample_path)

    index_path = analysis_dir / "analysis_index.jsonl"
    index_record = {
        "sample_id": sample_id,
        "sample_file_id": sample_file_id,
        "question_id": metadata.get("question_id"),
        "dataset_name": metadata.get("dataset_name"),
        "model_name": metadata.get("model_name"),
        "final_prediction_correct": metadata.get("final_prediction_correct"),
        "is_hallucinated": metadata.get("is_hallucinated"),
        "generated_answer": metadata.get("generated_answer"),
        "ground_truth_answer": metadata.get("ground_truth_answer"),
        "log_path": str(sample_path),
    }
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    return sample_path