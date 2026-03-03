import json

from pathlib import Path
from types import SimpleNamespace

from swebench.harness.apptainer_eval import (
    APPLY_FAIL_EXIT_CODE,
    run_instance_apptainer,
)
from swebench.harness.apptainer_utils import ApptainerExecResult
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness import run_evaluation as run_evaluation_module


def _pred(instance_id: str) -> dict:
    return {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: "test/model",
        KEY_PREDICTION: "diff --git a/a b/a\n",
    }


def _spec(instance_id: str):
    return SimpleNamespace(
        instance_id=instance_id,
        instance_image_key=f"swebench/sweb.eval.x86_64.{instance_id}:latest",
        eval_script="echo hello from eval",
        docker_specs={},
    )


def test_run_instance_apptainer_happy_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _spec("sympy__sympy-20590")
    pred = _pred(spec.instance_id)

    monkeypatch.setattr(
        "swebench.harness.apptainer_eval.ensure_apptainer_image",
        lambda **kwargs: Path("/tmp/fake.sif"),
    )
    monkeypatch.setattr(
        "swebench.harness.apptainer_eval.run_apptainer_exec",
        lambda **kwargs: ApptainerExecResult(
            output="orchestrate logs", return_code=0, timed_out=False, runtime=1.0
        ),
    )
    monkeypatch.setattr(
        "swebench.harness.apptainer_eval.get_eval_report",
        lambda **kwargs: {spec.instance_id: {"resolved": True}},
    )

    result = run_instance_apptainer(spec, pred, run_id="apptainer-test", timeout=60)

    assert result == {"completed": True, "resolved": True}
    report_path = (
        tmp_path
        / "logs"
        / "run_evaluation"
        / "apptainer-test"
        / "test__model"
        / spec.instance_id
        / "report.json"
    )
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report[spec.instance_id]["resolved"] is True


def test_run_instance_apptainer_patch_apply_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _spec("chartjs__chartjs-1234")
    pred = _pred(spec.instance_id)

    monkeypatch.setattr(
        "swebench.harness.apptainer_eval.ensure_apptainer_image",
        lambda **kwargs: Path("/tmp/fake.sif"),
    )
    monkeypatch.setattr(
        "swebench.harness.apptainer_eval.run_apptainer_exec",
        lambda **kwargs: ApptainerExecResult(
            output="patch apply failed",
            return_code=APPLY_FAIL_EXIT_CODE,
            timed_out=False,
            runtime=1.0,
        ),
    )

    result = run_instance_apptainer(spec, pred, run_id="apptainer-test", timeout=60)
    assert result == {"completed": False, "resolved": False}


def test_main_dispatches_to_apptainer_runtime(monkeypatch):
    instance = {
        KEY_INSTANCE_ID: "sympy__sympy-20590",
        "repo": "sympy/sympy",
        "version": "1.12",
        "base_commit": "abc123",
        "patch": "",
        "test_patch": "",
        "problem_statement": "",
        "hints_text": "",
        "created_at": "",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "",
    }
    pred = _pred(instance[KEY_INSTANCE_ID])

    called = {}

    monkeypatch.setattr(
        run_evaluation_module,
        "get_predictions_from_file",
        lambda *args, **kwargs: [pred],
    )
    monkeypatch.setattr(
        run_evaluation_module,
        "get_dataset_from_preds",
        lambda *args, **kwargs: [instance],
    )
    monkeypatch.setattr(
        run_evaluation_module,
        "load_swebench_dataset",
        lambda *args, **kwargs: [instance],
    )
    monkeypatch.setattr(
        run_evaluation_module,
        "make_test_spec",
        lambda *args, **kwargs: _spec(instance[KEY_INSTANCE_ID]),
    )
    monkeypatch.setattr(
        run_evaluation_module,
        "validate_apptainer_runtime_config",
        lambda **kwargs: called.setdefault("validated", True),
    )
    monkeypatch.setattr(
        run_evaluation_module,
        "run_instances_apptainer",
        lambda **kwargs: called.setdefault("run", kwargs),
    )
    monkeypatch.setattr(
        run_evaluation_module,
        "make_run_report",
        lambda *args, **kwargs: Path("apptainer-report.json"),
    )
    monkeypatch.setattr(run_evaluation_module.platform, "system", lambda: "Darwin")

    out = run_evaluation_module.main(
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        instance_ids=[instance[KEY_INSTANCE_ID]],
        predictions_path="dummy.jsonl",
        max_workers=1,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id="apptainer-run",
        timeout=30,
        namespace="swebench",
        rewrite_reports=False,
        modal=False,
        runtime="apptainer",
        instance_image_tag="latest",
        env_image_tag="latest",
        report_dir=".",
    )

    assert called["validated"] is True
    assert "run" in called
    assert called["run"]["run_id"] == "apptainer-run"
    assert out == Path("apptainer-report.json")
