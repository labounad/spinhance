"""
Tests for simulation.mnova_runner pure helpers (no MestReNova required).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.mnova_runner import _shard, _app_bundle, _launch_cmd  # noqa: E402


def test_shard_partitions_all_items():
    items = list(range(100))
    shards = _shard(items, 8)
    assert len(shards) == 8
    # Every item appears exactly once, nothing lost or duplicated.
    flat = sorted(x for s in shards for x in s)
    assert flat == items


def test_shard_balanced_round_robin():
    shards = _shard(list(range(100)), 8)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1  # balanced to within one item


def test_shard_caps_workers_at_items():
    shards = _shard([1, 2, 3], 8)
    assert len(shards) == 3
    assert sorted(x for s in shards for x in s) == [1, 2, 3]


def test_app_bundle_extraction():
    exe = Path("/Applications/MestReNova.app/Contents/MacOS/MestReNova")
    assert _app_bundle(exe) == "/Applications/MestReNova.app"


def test_launch_cmd_open():
    cmd = _launch_cmd(Path("/Applications/MestReNova.app/Contents/MacOS/MestReNova"),
                      "open", Path("/x"), Path("/y"))
    assert cmd[:3] == ["open", "-na", "/Applications/MestReNova.app"]
    assert cmd[-3:] == ["--args", "-sf", "spinhanceBatch,/x,/y"]


def test_launch_cmd_direct():
    cmd = _launch_cmd(Path("/bin/MestReNova"), "direct", Path("/x"), Path("/y"))
    assert cmd == ["/bin/MestReNova", "-sf", "spinhanceBatch,/x,/y"]


def test_launch_cmd_invalid():
    import pytest
    with pytest.raises(ValueError):
        _launch_cmd(Path("/bin/x"), "bogus", Path("/x"), Path("/y"))
