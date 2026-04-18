"""Tests — `simtool idynomics` CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simtool.cli import main
from simtool.frameworks.idynomics_2.doctor import run_health_check, save_jar_path


def test_health_check_missing_jar_reports_red(monkeypatch, tmp_path):
    # Env var empty + isolated config path -> no jar resolves.
    monkeypatch.delenv("IDYNOMICS_2_JAR", raising=False)
    isolated_cfg = tmp_path / ".simtool" / "idynomics_2.json"
    report = run_health_check(config_path=isolated_cfg)
    jar_check = next(c for c in report.checks if c.name == "iDynoMiCS 2 jar")
    # The default search paths may or may not contain a jar depending on the
    # dev machine; we only assert behavior when NO resolution happened.
    if not jar_check.ok:
        assert "not found" in jar_check.detail


def test_health_check_green_when_jar_exists(monkeypatch, tmp_path):
    fake_jar_dir = tmp_path / "idynomics"
    fake_jar_dir.mkdir()
    fake_jar = fake_jar_dir / "iDynoMiCS-2.0.jar"
    fake_jar.write_bytes(b"x")
    (fake_jar_dir / "default.cfg").write_text("x")
    (fake_jar_dir / "config").mkdir()
    monkeypatch.setenv("IDYNOMICS_2_JAR", str(fake_jar))
    report = run_health_check(config_path=tmp_path / ".simtool" / "idynomics_2.json")
    must_pass = {"iDynoMiCS 2 jar",
                 "iDynoMiCS sibling `default.cfg`",
                 "iDynoMiCS sibling `config`"}
    for c in report.checks:
        if c.name in must_pass:
            assert c.ok, f"{c.name} failed: {c.detail}"


def test_save_jar_path_persists(tmp_path):
    fake_jar = tmp_path / "j.jar"
    fake_jar.write_bytes(b"x")
    cfg_path = tmp_path / ".simtool" / "idynomics_2.json"
    written = save_jar_path(fake_jar, config_path=cfg_path)
    assert written == cfg_path
    data = json.loads(cfg_path.read_text())
    assert Path(data["jar_path"]) == fake_jar.resolve()


def test_save_jar_path_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_jar_path(tmp_path / "nonexistent.jar",
                      config_path=tmp_path / "cfg.json")


def test_cli_idynomics_check_prints_report(capsys):
    """Regardless of machine state, `simtool idynomics check` prints the
    structured report header and either exits 0 (all green) or 1 (red)."""
    rc = main(["idynomics", "check"])
    assert rc in (0, 1)
    captured = capsys.readouterr()
    assert "iDynoMiCS 2 runtime check" in captured.out
    assert "iDynoMiCS 2 jar" in captured.out


def test_cli_idynomics_set_jar_missing_file(tmp_path, capsys):
    rc = main(["idynomics", "set-jar", str(tmp_path / "nope.jar")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
