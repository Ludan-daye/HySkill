#!/usr/bin/env python3
"""Orchestrate one model's complete E7 hy_rerank lane (decisions, answers, eval, audit).

Starts ``vllm serve`` with the frozen K=2 parameters, reads live ``/v1/models``
and GPU/software facts, validates exact match with ``FROZEN_K2_RUNTIME_REFERENCES``,
runs a 20-record canary on the first domain before completing all four rule
domains, evaluates answers, and audits every domain to ``valid=true``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    FROZEN_SRAGENTS_REVISION,
    JsonObject,
    canonical_json,
    sha256_file,
    validate_frozen_k2_runtime_reference,
    write_json_atomic,
)
from hyskill.runtime_matched_rerank import (
    HY_RERANK_ARM,
    answer_generation,
    load_native_rerank_runtime,
    read_jsonl,
    rerank_generation,
)

DOMAINS_RUN_ORDER: tuple[tuple[str, int], ...] = (
    ("champ", 223),
    ("theoremqa", 747),
    ("logicbench", 760),
    ("medcalcbench", 1100),
)

DECISION_CODE_FILES: tuple[str, ...] = (
    "external/SR-Agents/src/sragents/corpus.py",
    "external/SR-Agents/src/sragents/llm.py",
    "external/SR-Agents/src/sragents/prompts.py",
    "external/SR-Agents/src/sragents/retrieve/llm_rerank.py",
    "hyskill/runtime_matched_execution.py",
    "hyskill/runtime_matched_rerank.py",
    "scripts/run_runtime_matched_rerank_decisions.py",
)

ANSWER_CODE_FILES: tuple[str, ...] = (
    "external/SR-Agents/src/sragents/infer/base.py",
    "external/SR-Agents/src/sragents/infer/engines/direct.py",
    "external/SR-Agents/src/sragents/infer/engines/tool_loop.py",
    "external/SR-Agents/src/sragents/llm.py",
    "external/SR-Agents/src/sragents/prompts.py",
    "hyskill/runtime_matched_execution.py",
    "hyskill/runtime_matched_rerank.py",
    "scripts/run_runtime_matched_rerank_answers.py",
)

CHECKPOINT_DIR_NAMES: dict[str, str] = {
    "glm4-9b": "glm4-9b",
    "llama31-8b": "llama31-8b",
    "mistral7b": "mistral7b",
    "qwen3.5-4b-reference": "qwen3.5-4b",
    "qwen35-9b": "qwen35-9b",
}


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def _query_gpu_and_software(vllm_python: Path) -> tuple[dict[str, str], dict[str, str]]:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    parts = [p.strip() for p in smi.split(",")]
    gpu_model, gpu_uuid, driver_version = parts[0], parts[1], parts[2]

    probe_code = (
        "import json, platform, torch, transformers, vllm; "
        "print(json.dumps({"
        "'python_version': platform.python_version(), "
        "'pytorch_version': torch.__version__, "
        "'transformers_version': transformers.__version__, "
        "'cuda_version': str(torch.version.cuda), "
        "'vllm_version': vllm.__version__"
        "}))"
    )
    sw_raw = subprocess.run(
        [str(vllm_python), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
    sw = cast(dict[str, str], json.loads(sw_raw))
    sw["driver_version"] = driver_version
    hw = {"gpu_model": gpu_model, "gpu_uuid": gpu_uuid}
    return hw, sw


def _verify_local_tokenizer_files(checkpoint_dir: Path, model_tag: str) -> dict[str, str]:
    ref = FROZEN_K2_RUNTIME_REFERENCES[model_tag]
    observed: dict[str, str] = {}
    for fname, expected_sha in ref["tokenizer_artifacts"].items():
        fpath = checkpoint_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"missing tokenizer file {fpath}")
        actual_sha = sha256_file(fpath)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"tokenizer SHA mismatch for {model_tag}/{fname}: "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        observed[fname] = actual_sha
    return observed


def _wait_for_vllm(api_base: str, proc: subprocess.Popen[bytes], timeout_s: int = 420) -> JsonObject:
    url = f"{api_base}/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited prematurely with code {proc.returncode}")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
                        return cast(JsonObject, data)
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for vLLM at {url}")


def _build_manifest(
    python_exe: Path,
    repo_root: Path,
    runtime_facts_path: Path,
    generation_path: Path,
    artifacts: list[tuple[str, Path]],
    code_files: tuple[str, ...],
    output_path: Path,
    env: dict[str, str],
) -> None:
    cmd = [
        str(python_exe),
        str(repo_root / "scripts" / "build_runtime_matched_runtime_manifest.py"),
        "--runtime-facts",
        str(runtime_facts_path),
        "--generation",
        str(generation_path),
        "--repository-root",
        str(repo_root),
        "--output",
        str(output_path),
    ]
    for name, path in artifacts:
        cmd.extend(["--artifact", f"{name}={path}"])
    for cf in code_files:
        cmd.extend(["--code-file", cf])
    subprocess.run(cmd, check=True, env=env)


def _run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n--- LOG TAIL ---\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E7 hy_rerank lane for one model.")
    parser.add_argument("--model-tag", required=True, choices=tuple(CHECKPOINT_DIR_NAMES))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--vllm-venv", required=True, type=Path)
    parser.add_argument("--client-python", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.72)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    model_tag: str = str(args.model_tag)
    ref = FROZEN_K2_RUNTIME_REFERENCES[model_tag]
    served_model: str = ref["served_model"]
    repo_root: Path = args.repo_root.resolve()
    model_out: Path = (args.result_root / model_tag).resolve()
    for sub in ("runtime", "decisions", "answers", "eval", "audits", "logs"):
        (model_out / sub).mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path = (args.models_root / CHECKPOINT_DIR_NAMES[model_tag]).resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint directory missing: {checkpoint_dir}")
    tok_artifacts = _verify_local_tokenizer_files(checkpoint_dir, model_tag)

    # Keep symlink paths intact so Python discovers pyvenv.cfg inside the venv!
    vllm_python = (args.vllm_venv.expanduser().absolute() / "bin" / "python")
    vllm_bin = (args.vllm_venv.expanduser().absolute() / "bin" / "vllm")
    client_python = args.client_python.expanduser().absolute()

    hw, sw = _query_gpu_and_software(vllm_python)
    if sw["vllm_version"] != ref["vllm_version"]:
        raise RuntimeError(
            f"vLLM version mismatch for {model_tag}: expected={ref['vllm_version']}, actual={sw['vllm_version']}"
        )

    sragents_checkout = repo_root / "external" / "SR-Agents"
    corpus_path = sragents_checkout / "data" / "bench" / "corpus" / "corpus.json"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus missing: {corpus_path}")
    native_rt = load_native_rerank_runtime(sragents_checkout, FROZEN_SRAGENTS_REVISION)
    extra_body = native_rt["get_extra_body"](served_model, False)
    dec_gen = rerank_generation(extra_body)
    ans_gen = answer_generation(extra_body)

    api_base = f"http://127.0.0.1:{args.port}/v1"
    vllm_cmd = [
        str(vllm_python),
        str(vllm_bin),
        "serve",
        str(checkpoint_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-model-len",
        "8192",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--served-model-name",
        served_model,
        "--gpu-memory-utilization",
        str(args.gpu_mem_util),
    ]
    if model_tag == "glm4-9b":
        vllm_cmd.append("--trust-remote-code")

    vllm_log_path = model_out / "logs" / "vllm_server.log"
    _log(f"Starting vLLM for {model_tag} ({served_model}) -> {vllm_log_path}")
    vllm_env = os.environ.copy()
    vllm_env["NO_PROXY"] = "127.0.0.1,localhost"
    vllm_env["no_proxy"] = "127.0.0.1,localhost"
    vllm_env["HF_HUB_OFFLINE"] = "1"
    vllm_env["TRANSFORMERS_OFFLINE"] = "1"

    vllm_log_fh = vllm_log_path.open("w", encoding="utf-8")
    vllm_proc = subprocess.Popen(
        vllm_cmd,
        stdout=vllm_log_fh,
        stderr=subprocess.STDOUT,
        env=vllm_env,
        start_new_session=True,
    )

    try:
        models_readback = _wait_for_vllm(api_base, vllm_proc, timeout_s=480)
        _log(f"vLLM ready for {model_tag}: readback={models_readback}")

        client_env = os.environ.copy()
        client_env["PYTHONPATH"] = f"{repo_root}:{sragents_checkout / 'src'}"
        client_env["NO_PROXY"] = "127.0.0.1,localhost"
        client_env["no_proxy"] = "127.0.0.1,localhost"

        def make_facts(domain: str, stage: str) -> JsonObject:
            facts: JsonObject = {
                "schema_version": "runtime-matched-runtime-facts-v1",
                "job": {
                    "job_id": f"{model_tag}-{domain}-{HY_RERANK_ARM}-{stage}-20261125-v1",
                    "result_tag": model_tag,
                    "model": served_model,
                    "domain": domain,
                    "arm": HY_RERANK_ARM,
                    "stage": stage,
                },
                "checkpoint": {
                    "repository": ref["checkpoint_repository"],
                    "revision": ref["checkpoint_revision"],
                    "path": str(checkpoint_dir),
                    "provenance": "China mirror checkpoint verified against frozen K=2 tokenizer and revision identity",
                    "files_manifest_sha256": ref["checkpoint_files_manifest_sha256"],
                },
                "tokenizer": {
                    "artifacts": dict(tok_artifacts),
                    "chat_template_sha256": ref["chat_template_sha256"],
                },
                "endpoint": {
                    "api_base": api_base,
                    "served_model": served_model,
                    "process_command": " ".join(vllm_cmd),
                    "vllm_version": sw["vllm_version"],
                    "dtype": "bfloat16",
                    "quantization": "none",
                    "max_model_len": 8192,
                    "tensor_parallel_size": 1,
                    "models_readback": models_readback,
                },
                "software": dict(sw),
                "hardware": dict(hw),
                "source": {
                    "sr_agents_revision": FROZEN_SRAGENTS_REVISION,
                },
            }
            validate_frozen_k2_runtime_reference(facts)
            return facts

        for dom_idx, (domain, expected_count) in enumerate(DOMAINS_RUN_ORDER):
            _log(f"=== {model_tag} :: {domain} (expected={expected_count}) ===")
            audit_out = model_out / "audits" / f"{domain}-{HY_RERANK_ARM}.audit.json"
            if audit_out.is_file():
                existing_audit = json.loads(audit_out.read_text(encoding="utf-8"))
                if existing_audit.get("valid") is True:
                    _log(f"Domain {model_tag}:{domain} already audited valid=True; skipping.")
                    continue

            instances_path = sragents_checkout / "data" / "bench" / "instances" / f"{domain}.json"
            cand_path = repo_root / "results" / "hy-rerank-inputs" / "candidates" / model_tag / f"{domain}-routed_k2.json"
            val_path = repo_root / "results" / "hy-rerank-inputs" / "validation" / model_tag / f"{domain}.val_ids.json"

            dec_facts_path = model_out / "runtime" / f"{domain}-{HY_RERANK_ARM}-decision.runtime-facts.json"
            dec_gen_path = model_out / "runtime" / f"{domain}-{HY_RERANK_ARM}-decision.generation.json"
            dec_manifest_path = model_out / "runtime" / f"{domain}-{HY_RERANK_ARM}-decision.manifest.json"
            if not dec_manifest_path.is_file():
                write_json_atomic(dec_facts_path, make_facts(domain, "decision"))
                write_json_atomic(dec_gen_path, dec_gen)
                _build_manifest(
                    client_python,
                    repo_root,
                    dec_facts_path,
                    dec_gen_path,
                    [
                        ("routed_candidates", cand_path),
                        ("corpus", corpus_path),
                        ("instances", instances_path),
                    ],
                    DECISION_CODE_FILES,
                    dec_manifest_path,
                    client_env,
                )

            dec_out = model_out / "decisions" / f"{domain}-{HY_RERANK_ARM}.decisions.jsonl"
            dec_attempt_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.decision.attempts.jsonl"
            dec_canary_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.decision.canary.log"
            dec_full_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.decision.full.log"

            base_dec_cmd = [
                str(client_python),
                str(repo_root / "scripts" / "run_runtime_matched_rerank_decisions.py"),
                "--instances",
                str(instances_path),
                "--corpus",
                str(corpus_path),
                "--candidate-source",
                str(cand_path),
                "--arm",
                HY_RERANK_ARM,
                "--output",
                str(dec_out),
                "--attempt-log",
                str(dec_attempt_log),
                "--runtime-manifest",
                str(dec_manifest_path),
                "--result-tag",
                model_tag,
                "--model",
                served_model,
                "--api-base",
                api_base,
                "--domain",
                domain,
                "--workers",
                str(args.workers),
                "--sragents-checkout",
                str(sragents_checkout),
                "--repository-root",
                str(repo_root),
                "--sragents-revision",
                FROZEN_SRAGENTS_REVISION,
            ]

            if dom_idx == 0 and not dec_out.is_file():
                _log(f"Running 20-record decision canary for {model_tag}:{domain}...")
                _run_logged([*base_dec_cmd, "--max-new-records", "20"], dec_canary_log, client_env)
                canary_rows = read_jsonl(dec_out)
                if len(canary_rows) != 20:
                    raise RuntimeError(f"Expected 20 canary rows, got {len(canary_rows)}")
                for r in canary_rows:
                    if r.get("arm") != HY_RERANK_ARM:
                        raise RuntimeError(f"Bad canary arm: {r.get('arm')}")
                    if not isinstance(r.get("omitted_candidate_count"), int):
                        raise RuntimeError(f"Missing omitted_candidate_count in canary: {r}")
                    if r.get("failure_category") not in ("success", "method_failure"):
                        raise RuntimeError(f"Unresolved canary category: {r.get('failure_category')}")
                mean_omitted = sum(int(r["omitted_candidate_count"]) for r in canary_rows) / 20.0  # type: ignore[arg-type]
                _log(f"CANARY_VERIFIED {model_tag}:{domain} rows=20 mean_omitted={mean_omitted:.2f}")

            _log(f"Running full decisions for {model_tag}:{domain}...")
            _run_logged([*base_dec_cmd, "--max-new-records", "0"], dec_full_log, client_env)
            _log(f"Decisions complete for {model_tag}:{domain}")

            ans_facts_path = model_out / "runtime" / f"{domain}-{HY_RERANK_ARM}-answer.runtime-facts.json"
            ans_gen_path = model_out / "runtime" / f"{domain}-{HY_RERANK_ARM}-answer.generation.json"
            ans_manifest_path = model_out / "runtime" / f"{domain}-{HY_RERANK_ARM}-answer.manifest.json"
            ans_out = model_out / "answers" / f"{domain}-{HY_RERANK_ARM}.jsonl"
            # A new vLLM process rewrites models_readback and changes the
            # manifest digest. Existing answer rows are bound to the digest
            # they were generated under, so a partial file must keep it.
            if ans_manifest_path.is_file() and ans_out.is_file() and ans_out.stat().st_size > 0:
                _log(f"Reusing existing answer manifest for {model_tag}:{domain}")
            else:
                write_json_atomic(ans_facts_path, make_facts(domain, "answer"))
                write_json_atomic(ans_gen_path, ans_gen)
                _build_manifest(
                    client_python,
                    repo_root,
                    ans_facts_path,
                    ans_gen_path,
                    [
                        ("corpus", corpus_path),
                        ("rerank_decisions", dec_out),
                        ("instances", instances_path),
                    ],
                    ANSWER_CODE_FILES,
                    ans_manifest_path,
                    client_env,
                )
            ans_attempt_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.answer.attempts.jsonl"
            ans_full_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.answer.full.log"
            ans_cmd = [
                str(client_python),
                str(repo_root / "scripts" / "run_runtime_matched_rerank_answers.py"),
                "--instances",
                str(instances_path),
                "--corpus",
                str(corpus_path),
                "--decisions",
                str(dec_out),
                "--arm",
                HY_RERANK_ARM,
                "--output",
                str(ans_out),
                "--attempt-log",
                str(ans_attempt_log),
                "--runtime-manifest",
                str(ans_manifest_path),
                "--result-tag",
                model_tag,
                "--model",
                served_model,
                "--api-base",
                api_base,
                "--domain",
                domain,
                "--workers",
                str(args.workers),
                "--max-new-records",
                "0",
                "--sragents-checkout",
                str(sragents_checkout),
                "--repository-root",
                str(repo_root),
                "--sragents-revision",
                FROZEN_SRAGENTS_REVISION,
            ]
            _log(f"Running full answers for {model_tag}:{domain}...")
            _run_logged(ans_cmd, ans_full_log, client_env)
            _log(f"Answers complete for {model_tag}:{domain}")

            eval_out = model_out / "eval" / f"{domain}-{HY_RERANK_ARM}.eval.json"
            eval_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.eval.log"
            eval_cmd = [
                str(client_python),
                str(repo_root / "scripts" / "evaluate_runtime_matched_baselines.py"),
                "--answers",
                str(ans_out),
                "--instances",
                str(instances_path),
                "--validation-source",
                str(val_path),
                "--result-tag",
                model_tag,
                "--served-model",
                served_model,
                "--domain",
                domain,
                "--arm",
                HY_RERANK_ARM,
                "--expected-count",
                str(expected_count),
                "--output",
                str(eval_out),
            ]
            _log(f"Evaluating {model_tag}:{domain}...")
            _run_logged(eval_cmd, eval_log, client_env)

            audit_log = model_out / "logs" / f"{domain}-{HY_RERANK_ARM}.audit.log"
            audit_cmd = [
                str(client_python),
                str(repo_root / "scripts" / "audit_runtime_matched_native_domain.py"),
                "--instances",
                str(instances_path),
                "--decisions",
                str(dec_out),
                "--answers",
                str(ans_out),
                "--decision-attempt-log",
                str(dec_attempt_log),
                "--answer-attempt-log",
                str(ans_attempt_log),
                "--decision-full-log",
                str(dec_full_log),
                "--answer-full-log",
                str(ans_full_log),
                "--decision-manifest",
                str(dec_manifest_path),
                "--answer-manifest",
                str(ans_manifest_path),
                "--evaluation",
                str(eval_out),
                "--repository-root",
                str(repo_root),
                "--result-tag",
                model_tag,
                "--served-model",
                served_model,
                "--domain",
                domain,
                "--arm",
                HY_RERANK_ARM,
                "--expected-count",
                str(expected_count),
                "--output",
                str(audit_out),
            ]
            _log(f"Auditing {model_tag}:{domain}...")
            _run_logged(audit_cmd, audit_log, client_env)
            audit_data = json.loads(audit_out.read_text(encoding="utf-8"))
            if audit_data.get("valid") is not True:
                raise RuntimeError(f"Audit valid!=True for {model_tag}:{domain}")
            heldout_acc = audit_data["answer_metrics"]["heldout"]["accuracy"]
            _log(f"AUDIT_VALID {model_tag}:{domain} heldout_acc={heldout_acc:.4f}")

        _log(f"MODEL_COMPLETE {model_tag} ALL_4_DOMAINS_VALID")
    finally:
        _log(f"Stopping vLLM process group for {model_tag} (pid={vllm_proc.pid})...")
        try:
            os.killpg(os.getpgid(vllm_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            vllm_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(vllm_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        vllm_log_fh.close()


if __name__ == "__main__":
    main()
