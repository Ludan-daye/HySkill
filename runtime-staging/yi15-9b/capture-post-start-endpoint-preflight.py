#!/usr/bin/env python3
"""Capture immutable post-start evidence for the Yi baseline endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias, cast


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
CHECKPOINT_ARTIFACTS: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)
SENSITIVE_VALUE_PATTERN: re.Pattern[str] = re.compile(
    r"(?i)\b(?:api[_-]?key|authorization|bearer|password|passwd|secret|"
    r"credential|access[_-]?token|refresh[_-]?token|private[_-]?key)\b"
)


class EndpointPreflightError(RuntimeError):
    """Raised when live endpoint evidence is missing or inconsistent."""


def parse_args() -> argparse.Namespace:
    """Parse explicit evidence inputs and expected endpoint identity."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-repository", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-files-manifest-sha256", required=True)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--models-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--required-environment-name", required=True)
    parser.add_argument("--required-environment-value", required=True)
    parser.add_argument("--http-timeout-seconds", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of one byte string."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one required regular file without loading it into memory."""

    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: path={path}")
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, context: str) -> str:
    """Return one lowercase SHA-256 digest or raise."""

    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EndpointPreflightError(
            f"Expected lowercase SHA-256: context={context}, value={value!r}"
        )
    return value


def load_pid(path: Path) -> int:
    """Read one live positive PID from a required file."""

    if not path.is_file():
        raise FileNotFoundError(f"Endpoint PID file does not exist: path={path}")
    raw_value: str = path.read_text(encoding="utf-8").strip()
    try:
        pid: int = int(raw_value)
    except ValueError as error:
        raise EndpointPreflightError(
            f"Endpoint PID file is not an integer: path={path}, value={raw_value!r}"
        ) from error
    if pid <= 0 or not Path(f"/proc/{pid}").is_dir():
        raise EndpointPreflightError(
            f"Endpoint process is not live: path={path}, pid={pid}"
        )
    return pid


def read_cmdline(pid: int) -> list[str]:
    """Read the exact non-empty process argument vector."""

    path = Path(f"/proc/{pid}/cmdline")
    values: list[str] = [
        item.decode("utf-8")
        for item in path.read_bytes().split(b"\0")
        if item
    ]
    if not values:
        raise EndpointPreflightError(
            f"Endpoint process command line is empty: pid={pid}"
        )
    return values


def read_required_environment(
    pid: int,
    variable_name: str,
    expected_value: str,
) -> JsonObject:
    """Read only the one required long-context environment fact."""

    path = Path(f"/proc/{pid}/environ")
    environment: dict[str, str] = {}
    for item in path.read_bytes().split(b"\0"):
        if not item:
            continue
        raw_name, separator, raw_value = item.partition(b"=")
        if separator != b"=":
            raise EndpointPreflightError(
                f"Malformed endpoint environment entry: pid={pid}"
            )
        name: str = raw_name.decode("utf-8")
        if name == variable_name:
            environment[name] = raw_value.decode("utf-8")
    actual_value: str | None = environment.get(variable_name)
    if actual_value != expected_value:
        raise EndpointPreflightError(
            "Required endpoint environment mismatch: "
            f"pid={pid}, name={variable_name}, "
            f"expected={expected_value!r}, actual={actual_value!r}"
        )
    return {
        variable_name: actual_value,
    }


def load_models_readback(
    models_url: str,
    timeout_seconds: float,
    served_model: str,
    expected_max_model_len: int,
    expected_checkpoint: Path,
) -> tuple[JsonObject, str]:
    """Fetch and validate the live model identity and 8K readback."""

    with urllib.request.urlopen(
        models_url,
        timeout=timeout_seconds,
    ) as response:
        response_bytes: bytes = response.read()
    try:
        payload: JsonValue = cast(
            JsonValue,
            json.loads(response_bytes.decode("utf-8")),
        )
    except json.JSONDecodeError as error:
        raise EndpointPreflightError(
            "Endpoint model readback is malformed JSON: "
            f"url={models_url}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise EndpointPreflightError(
            f"Endpoint model readback is not an object: url={models_url}"
        )
    raw_data: JsonValue | None = payload.get("data")
    if not isinstance(raw_data, list):
        raise EndpointPreflightError(
            f"Endpoint model readback has no data list: url={models_url}"
        )
    matching_models: list[JsonObject] = [
        item
        for item in raw_data
        if isinstance(item, dict) and item.get("id") == served_model
    ]
    if len(matching_models) != 1:
        raise EndpointPreflightError(
            "Endpoint served-model readback is not unique: "
            f"url={models_url}, model={served_model}, "
            f"matches={len(matching_models)}"
        )
    model: JsonObject = matching_models[0]
    actual_max_model_len: JsonValue | None = model.get("max_model_len")
    if actual_max_model_len != expected_max_model_len:
        raise EndpointPreflightError(
            "Endpoint max-model-len mismatch: "
            f"model={served_model}, expected={expected_max_model_len}, "
            f"actual={actual_max_model_len!r}"
        )
    actual_checkpoint: JsonValue | None = model.get("root")
    if actual_checkpoint != str(expected_checkpoint):
        raise EndpointPreflightError(
            "Endpoint checkpoint path mismatch: "
            f"model={served_model}, expected={expected_checkpoint}, "
            f"actual={actual_checkpoint!r}"
        )
    selected_readback: JsonObject = {
        "url": models_url,
        "served_model": served_model,
        "max_model_len": expected_max_model_len,
        "root": actual_checkpoint,
        "object": model.get("object"),
        "owned_by": model.get("owned_by"),
        "response_sha256": sha256_bytes(response_bytes),
    }
    return selected_readback, response_bytes.decode("utf-8")


def load_gpu_identity() -> JsonObject:
    """Read the single visible GPU model, UUID, and driver."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[str] = [
        row.strip()
        for row in completed.stdout.splitlines()
        if row.strip()
    ]
    if len(rows) != 1:
        raise EndpointPreflightError(
            f"Expected exactly one visible GPU: rows={rows}"
        )
    columns: list[str] = [
        column.strip()
        for column in rows[0].split(",")
    ]
    if len(columns) != 3 or any(not column for column in columns):
        raise EndpointPreflightError(
            f"Malformed nvidia-smi identity row: row={rows[0]!r}"
        )
    return {
        "gpu_model": columns[0],
        "gpu_uuid": columns[1],
        "driver_version": columns[2],
    }


def build_checkpoint_identity(
    checkpoint: Path,
    repository: str,
    revision: str,
    files_manifest_sha256: str,
) -> JsonObject:
    """Hash the exact tokenizer and configuration artifacts."""

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: path={checkpoint}"
        )
    artifact_hashes: JsonObject = {
        name: sha256_file(checkpoint / name)
        for name in CHECKPOINT_ARTIFACTS
    }
    return {
        "repository": repository,
        "revision": revision,
        "path": str(checkpoint),
        "files_manifest_sha256": require_sha256(
            files_manifest_sha256,
            "checkpoint-files-manifest-sha256",
        ),
        "artifact_sha256": artifact_hashes,
        "chat_template_sha256": artifact_hashes["tokenizer_config.json"],
    }


def bind_runtime_manifests(result_root: Path) -> JsonObject:
    """Bind all four pre-existing job manifests without changing them."""

    return {
        domain: {
            "path": str(
                result_root / "runtime" / f"{domain}-bare.manifest.json"
            ),
            "sha256": sha256_file(
                result_root / "runtime" / f"{domain}-bare.manifest.json"
            ),
        }
        for domain in DOMAINS
    }


def scan_sensitive_values(values: dict[str, str]) -> JsonObject:
    """Reject recognizable credential material in captured field values."""

    matches: list[str] = [
        field_path
        for field_path, value in values.items()
        if SENSITIVE_VALUE_PATTERN.search(value) is not None
    ]
    if matches:
        raise EndpointPreflightError(
            f"Sensitive-information scan failed: fields={matches}"
        )
    return {
        "scanner": "captured-field-values-v1",
        "status": "passed",
        "scanned_field_count": len(values),
        "match_count": 0,
    }


def write_new_file(path: Path, content: str) -> None:
    """Write one new file and refuse to replace existing evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
    except FileExistsError as error:
        raise EndpointPreflightError(
            f"Refusing to overwrite existing evidence: path={path}"
        ) from error


def canonical_json(payload: JsonObject) -> str:
    """Serialize one deterministic UTF-8 JSON document."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def main() -> None:
    """Capture, validate, scan, and hash the live endpoint evidence."""

    args = parse_args()
    result_root: Path = cast(Path, args.result_root).resolve()
    checkpoint: Path = cast(Path, args.checkpoint).resolve()
    pid_file: Path = cast(Path, args.pid_file).resolve()
    output: Path = cast(Path, args.output).resolve()
    pid: int = load_pid(pid_file)
    cmdline: list[str] = read_cmdline(pid)
    environment: JsonObject = read_required_environment(
        pid,
        cast(str, args.required_environment_name),
        cast(str, args.required_environment_value),
    )
    models_readback, raw_models_readback = load_models_readback(
        cast(str, args.models_url),
        cast(float, args.http_timeout_seconds),
        cast(str, args.served_model),
        cast(int, args.max_model_len),
        checkpoint,
    )
    hardware: JsonObject = load_gpu_identity()
    checkpoint_identity: JsonObject = build_checkpoint_identity(
        checkpoint,
        cast(str, args.checkpoint_repository),
        cast(str, args.checkpoint_revision),
        cast(str, args.checkpoint_files_manifest_sha256),
    )
    runtime_manifests: JsonObject = bind_runtime_manifests(result_root)
    sensitive_scan: JsonObject = scan_sensitive_values(
        {
            "endpoint.cmdline": "\0".join(cmdline),
            "endpoint.required_environment": json.dumps(
                environment,
                sort_keys=True,
            ),
            "endpoint.models_readback": raw_models_readback,
            "hardware": json.dumps(hardware, sort_keys=True),
            "checkpoint.path": str(checkpoint),
            "checkpoint.repository": cast(str, args.checkpoint_repository),
            "checkpoint.revision": cast(str, args.checkpoint_revision),
        }
    )
    payload: JsonObject = {
        "schema_version": "runtime-matched-post-start-endpoint-preflight-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": {
            "pid": pid,
            "cmdline": cmdline,
            "required_environment": environment,
            "models_readback": models_readback,
        },
        "hardware": hardware,
        "checkpoint": checkpoint_identity,
        "bound_runtime_manifests": runtime_manifests,
        "sensitive_information_scan": sensitive_scan,
    }
    write_new_file(output, canonical_json(payload))
    output_sha256: str = sha256_file(output)
    checksum_path = output.with_name(f"{output.name}.sha256")
    write_new_file(
        checksum_path,
        f"{output_sha256}  {output.name}\n",
    )
    print(
        json.dumps(
            {
                "event": "post_start_endpoint_preflight_captured",
                "output": str(output),
                "output_sha256": output_sha256,
                "checksum": str(checksum_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
