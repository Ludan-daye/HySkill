import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import pytest

from hyskill.downstream_reuse import JsonLike, JsonObject, JsonValue


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
PUBLISHED_PACK: Path = REPOSITORY_ROOT / "community-results" / "k2-fleet"
OUTPUT_FILENAMES: tuple[str, ...] = (
    "loading_metrics_long.jsonl.gz",
    "answer_metrics_long.jsonl.gz",
    "summary.json",
    "paired_comparisons.json",
    "manifest.json",
)


class InputPack(TypedDict):
    """Private aggregate inputs required by the fleet exporter."""

    loading_metrics_long: Path
    loading_summary: Path
    answer_metrics_long: Path
    answer_summary: Path
    paired_comparisons_current4: Path


def sha256_path(path: Path) -> str:
    """Return one file SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: JsonLike) -> None:
    """Write one deterministic fixture JSON file."""

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def source_entries(count: int, prefix: str) -> list[JsonObject]:
    """Build path-bearing private source identities."""

    return [
        {
            "path": f"/private/server/{prefix}-{index:03d}.json",
            "sha256": hashlib.sha256(
                f"{prefix}-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(count)
    ]


def build_private_inputs(root: Path) -> InputPack:
    """Reconstruct the validated private aggregates from the public pack."""

    root.mkdir(parents=True)
    loading_metrics_long: Path = root / "loading_metrics_long.jsonl"
    answer_metrics_long: Path = root / "answer_metrics_long.jsonl"
    loading_metrics_long.write_bytes(
        gzip.decompress(
            (PUBLISHED_PACK / "loading_metrics_long.jsonl.gz").read_bytes()
        )
    )
    answer_metrics_long.write_bytes(
        gzip.decompress(
            (PUBLISHED_PACK / "answer_metrics_long.jsonl.gz").read_bytes()
        )
    )
    public_summary: JsonObject = cast(
        JsonObject,
        json.loads(
            (PUBLISHED_PACK / "summary.json").read_text(encoding="utf-8")
        ),
    )
    loading_summary: JsonObject = copy.deepcopy(
        cast(JsonObject, public_summary["loading"])
    )
    loading_summary["input_files"] = source_entries(48, "loading")
    loading_summary["long_metrics_path"] = str(
        loading_metrics_long.resolve()
    )
    loading_summary["long_metrics_sha256"] = sha256_path(
        loading_metrics_long
    )
    answer_summary: JsonObject = copy.deepcopy(
        cast(JsonObject, public_summary["answers"])
    )
    answer_summary["inputs"] = source_entries(80, "answer")
    loading_summary_path: Path = root / "loading_summary.json"
    answer_summary_path: Path = root / "answer_summary.json"
    paired_path: Path = root / "paired_comparisons.current4.json"
    write_json(loading_summary_path, loading_summary)
    write_json(answer_summary_path, answer_summary)
    paired_path.write_bytes(
        (PUBLISHED_PACK / "paired_comparisons.json").read_bytes()
    )
    return {
        "loading_metrics_long": loading_metrics_long,
        "loading_summary": loading_summary_path,
        "answer_metrics_long": answer_metrics_long,
        "answer_summary": answer_summary_path,
        "paired_comparisons_current4": paired_path,
    }


def run_export(
    inputs: InputPack,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the fleet exporter as a real CLI process."""

    return subprocess.run(
        [
            sys.executable,
            "scripts/export_k2_fleet_pack.py",
            "--loading-metrics-long",
            str(inputs["loading_metrics_long"]),
            "--loading-summary",
            str(inputs["loading_summary"]),
            "--answer-metrics-long",
            str(inputs["answer_metrics_long"]),
            "--answer-summary",
            str(inputs["answer_summary"]),
            "--paired-comparisons-current4",
            str(inputs["paired_comparisons_current4"]),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def strings_in_json(value: JsonValue) -> list[str]:
    """Return every string value nested in one JSON value."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(strings_in_json(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(strings_in_json(item))
        return output
    return []


def test_rebuilds_published_pack_byte_for_byte_and_rejects_overwrite(
    tmp_path: Path,
) -> None:
    """Rebuild all five published files deterministically from private inputs."""

    inputs: InputPack = build_private_inputs(tmp_path / "inputs")
    output_dir: Path = tmp_path / "k2-fleet"
    result: subprocess.CompletedProcess[str] = run_export(inputs, output_dir)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output_dir.iterdir()} == set(
        OUTPUT_FILENAMES
    )
    for filename in OUTPUT_FILENAMES:
        assert (output_dir / filename).read_bytes() == (
            PUBLISHED_PACK / filename
        ).read_bytes()
    for filename in (
        "loading_metrics_long.jsonl.gz",
        "answer_metrics_long.jsonl.gz",
    ):
        assert (output_dir / filename).read_bytes()[4:8] == b"\x00" * 4
    summary: JsonValue = cast(
        JsonValue,
        json.loads((output_dir / "summary.json").read_text(encoding="utf-8")),
    )
    assert not any(
        value.startswith("/") for value in strings_in_json(summary)
    )
    manifest: JsonObject = cast(
        JsonObject,
        json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        ),
    )
    assert set(cast(JsonObject, manifest["files"])) == set(
        OUTPUT_FILENAMES
    ) - {"manifest.json"}
    assert manifest["manifest_self_policy"] == {
        "included_in_files": False,
        "reason": (
            "manifest.json is excluded because a file cannot contain its "
            "own stable SHA-256 digest."
        ),
    }

    overwrite_result: subprocess.CompletedProcess[str] = run_export(
        inputs,
        output_dir,
    )
    assert overwrite_result.returncode != 0
    assert "Refusing to overwrite" in overwrite_result.stderr


@pytest.mark.parametrize("corruption", ("unknown_comparison", "hash_mismatch"))
def test_rejects_unknown_comparisons_and_hash_mismatches_atomically(
    tmp_path: Path,
    corruption: str,
) -> None:
    """Reject baseline additions and broken source bindings before publishing."""

    inputs: InputPack = build_private_inputs(tmp_path / "inputs")
    if corruption == "unknown_comparison":
        paired: JsonObject = cast(
            JsonObject,
            json.loads(
                inputs["paired_comparisons_current4"].read_text(
                    encoding="utf-8"
                )
            ),
        )
        comparisons: JsonObject = cast(JsonObject, paired["comparisons"])
        comparisons["gated_vs_bare_seven_model"] = {}
        write_json(inputs["paired_comparisons_current4"], paired)
    elif corruption == "hash_mismatch":
        loading_summary: JsonObject = cast(
            JsonObject,
            json.loads(
                inputs["loading_summary"].read_text(encoding="utf-8")
            ),
        )
        loading_summary["long_metrics_sha256"] = "0" * 64
        write_json(inputs["loading_summary"], loading_summary)
    else:
        raise AssertionError(f"Unknown test corruption: value={corruption}")

    output_dir: Path = tmp_path / "rejected-pack"
    result: subprocess.CompletedProcess[str] = run_export(inputs, output_dir)
    assert result.returncode != 0
    assert not output_dir.exists()
