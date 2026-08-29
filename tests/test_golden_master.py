"""Golden-master regression test for pipeline_input.py.

Guards the refactor of pipeline_input.py: any structural change to it (or to
src/pipeline/**) must reproduce this exact per-frame JSON for a fixed, small
slice of one Waymo segment. Any diff here means behavior changed, not just
code structure -- refactors must keep this test green.

Skipped automatically when the local Waymo segment isn't present (e.g. in CI,
where the large .tfrecord fixtures are gitignored and not checked out).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENT = REPO_ROOT / "src/data/segment-1191788760630624072_3880_000_3900_000_with_camera_labels.tfrecord"
GOLDEN  = REPO_ROOT / "tests/golden" / (SEGMENT.stem + ".json")
MAX_FRAMES = 3


def _run_pipeline(output_dir: Path) -> None:
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "pipeline_input.py"),
            f"dataset.tfrecord_path={SEGMENT}",
            "dataset.segment_list=",
            "dataset.segment_dir=",
            f"dataset.max_frames={MAX_FRAMES}",
            "dataset.skip_existing=false",
            f"output.output_dir={output_dir}",
            "comet.api_key=",
            f"hydra.run.dir={output_dir / '_hydra'}",
            "hydra.job.chdir=False",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"pipeline_input.py exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.mark.skipif(not SEGMENT.exists(), reason="local Waymo segment fixture not present")
def test_pipeline_output_matches_golden_master(tmp_path):
    out_dir = tmp_path / "out"
    _run_pipeline(out_dir)

    produced = out_dir / (SEGMENT.stem + ".json")
    assert produced.exists(), f"expected output not written: {produced}"

    with open(produced) as f:
        actual = json.load(f)
    with open(GOLDEN) as f:
        expected = json.load(f)

    assert actual == expected, (
        "pipeline_input.py output no longer matches the golden master -- "
        "a refactor must not change behavior, only structure"
    )
