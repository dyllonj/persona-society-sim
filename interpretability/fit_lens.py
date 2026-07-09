from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import jlens
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Supports both `python -m` and direct script execution.
    from .common import load_prompts, parse_layers, sha256_file, sha256_json, write_json_atomic
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        load_prompts,
        parse_layers,
        sha256_file,
        sha256_json,
        write_json_atomic,
    )


JLENS_GIT_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and manifest a pinned Anthropic Jacobian Lens"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path)
    source.add_argument("--wikitext-prompts", type=int)
    parser.add_argument("--corpus-name", default="custom")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-layers", help="comma-separated zero-based decoder layers")
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts:
        prompts = load_prompts(args.prompts, limit=args.max_prompts)
        corpus_name = args.corpus_name
    else:
        from jlens.examples import load_wikitext_prompts

        count = args.wikitext_prompts
        if args.max_prompts is not None:
            count = min(count, args.max_prompts)
        prompts = load_wikitext_prompts(count)
        corpus_name = "wikitext-103-v1"

    corpus_records = [{"text": prompt} for prompt in prompts]
    corpus_path = args.output_dir / "fit_prompts.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in corpus_records),
        encoding="utf-8",
    )
    corpus_sha256 = sha256_file(corpus_path)

    dtype = _dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = jlens.from_hf(hf_model, tokenizer, compile=args.compile)
    source_layers = parse_layers(args.source_layers)
    checkpoint_path = args.output_dir / "fit.checkpoint.pt"

    started = time.monotonic()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    lens_path = args.output_dir / "lens.pt"
    lens.save(str(lens_path))

    resolved_model_revision = getattr(hf_model.config, "_commit_hash", None) or args.model_revision
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or resolved_model_revision
    lens_sha256 = sha256_file(lens_path)
    model_config = hf_model.config.to_dict()
    manifest = {
        "schema_version": "1.0",
        "lens_id": f"jlens-{lens_sha256[:16]}",
        "lens_sha256": lens_sha256,
        "jlens_git_commit": JLENS_GIT_COMMIT,
        "jlens_version": importlib.metadata.version("jlens"),
        "model_id": args.model_id,
        "model_revision": resolved_model_revision,
        "tokenizer_revision": tokenizer_revision,
        "model_config_sha256": sha256_json(model_config),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": args.dtype,
        "quantization": None,
        "d_model": lens.d_model,
        "n_layers": model.n_layers,
        "source_layers": lens.source_layers,
        "target_layer": args.target_layer if args.target_layer is not None else model.n_layers - 1,
        "corpus_name": corpus_name,
        "corpus_sha256": corpus_sha256,
        "corpus_path": corpus_path.name,
        "n_prompts_requested": len(prompts),
        "n_prompts": lens.n_prompts,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "dim_batch": args.dim_batch,
        "fit_seconds": time.monotonic() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
