from model_legacy.diagnostics import DiagnosticsWriter


def test_reset_live_files_clears_append_only_and_probe_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    writer = DiagnosticsWriter(run_dir)

    for name in (
        "metrics.jsonl",
        "events.jsonl",
        "system.jsonl",
        "status.json",
        "summary.json",
    ):
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n")

    probe_file = run_dir / "probes" / "epoch_0009" / "probe_metrics.json"
    probe_file.parent.mkdir(parents=True, exist_ok=True)
    probe_file.write_text("{}\n")

    ckpt_file = run_dir / "checkpoints" / "best.pt"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt_file.write_text("checkpoint placeholder\n")

    writer.reset_live_files()

    for name in (
        "metrics.jsonl",
        "events.jsonl",
        "system.jsonl",
        "status.json",
        "summary.json",
    ):
        assert not (run_dir / name).exists()

    assert not (run_dir / "probes").exists()

    # Checkpoints should not be deleted by diagnostics reset.
    assert ckpt_file.exists()
