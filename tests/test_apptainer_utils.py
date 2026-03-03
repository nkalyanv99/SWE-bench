import subprocess

from pathlib import Path

from swebench.harness.apptainer_utils import (
    build_apptainer_exec_cmd,
    ensure_apptainer_image,
    get_cached_sif_path,
    image_ref_to_apptainer_uri,
    image_ref_to_sif_name,
    run_apptainer_exec,
)


def test_image_ref_to_apptainer_uri_prefixes_docker():
    assert (
        image_ref_to_apptainer_uri("swebench/sweb.eval.x86_64.foo:latest")
        == "docker://swebench/sweb.eval.x86_64.foo:latest"
    )
    assert (
        image_ref_to_apptainer_uri("docker://swebench/sweb.eval.x86_64.foo:latest")
        == "docker://swebench/sweb.eval.x86_64.foo:latest"
    )


def test_image_ref_to_sif_name_is_deterministic():
    image_ref = "swebench/sweb.eval.x86_64.sympy__sympy-20590:latest"
    assert image_ref_to_sif_name(image_ref) == image_ref_to_sif_name(image_ref)
    assert image_ref_to_sif_name(image_ref).endswith(".sif")


def test_get_cached_sif_path_uses_cache_dir(tmp_path):
    image_ref = "swebench/sweb.eval.x86_64.foo:latest"
    path = get_cached_sif_path(image_ref, tmp_path)
    assert path.parent == tmp_path
    assert path.suffix == ".sif"


def test_ensure_apptainer_image_skips_pull_when_cached(tmp_path, monkeypatch):
    image_ref = "swebench/sweb.eval.x86_64.foo:latest"
    sif_path = get_cached_sif_path(image_ref, tmp_path)
    sif_path.write_text("cached")

    called = {"value": False}

    def _fake_run(*args, **kwargs):
        called["value"] = True
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr("swebench.harness.apptainer_utils.subprocess.run", _fake_run)
    result = ensure_apptainer_image(image_ref, tmp_path)
    assert result == sif_path
    assert called["value"] is False


def test_ensure_apptainer_image_pulls_when_missing(tmp_path, monkeypatch):
    image_ref = "swebench/sweb.eval.x86_64.foo:latest"
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        Path(cmd[2]).write_text("pulled")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("swebench.harness.apptainer_utils.subprocess.run", _fake_run)
    result = ensure_apptainer_image(image_ref, tmp_path)
    assert result.exists()
    assert len(calls) == 1
    assert calls[0][0][0:2] == ["apptainer", "pull"]
    assert calls[0][0][-1].startswith("docker://")


def test_build_apptainer_exec_cmd_renders_expected_command(tmp_path):
    sif_path = tmp_path / "image.sif"
    cmd = build_apptainer_exec_cmd(
        sif_path=sif_path,
        inner_cmd="echo hello",
        bind_mounts=[(tmp_path, "/host_tmp")],
        workdir="/testbed",
    )
    assert cmd[0:4] == ["apptainer", "exec", "--cleanenv", "--writable-tmpfs"]
    assert "--bind" in cmd
    assert "--pwd" in cmd
    assert str(sif_path) in cmd
    assert cmd[-3:] == ["/bin/bash", "-lc", "echo hello"]


def test_run_apptainer_exec_collects_output(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=7, stdout="stdout", stderr="stderr"
        )

    monkeypatch.setattr("swebench.harness.apptainer_utils.subprocess.run", _fake_run)
    result = run_apptainer_exec("/tmp/image.sif", "echo hi")
    assert result.return_code == 7
    assert result.timed_out is False
    assert result.output == "stdoutstderr"


def test_run_apptainer_exec_timeout(monkeypatch):
    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1, output="out", stderr="err")

    monkeypatch.setattr("swebench.harness.apptainer_utils.subprocess.run", _fake_run)
    result = run_apptainer_exec("/tmp/image.sif", "sleep 100", timeout=1)
    assert result.timed_out is True
    assert result.return_code == 124
    assert result.output == "outerr"
