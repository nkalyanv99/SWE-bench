from __future__ import annotations

import json
import threading
import traceback

from pathlib import Path

from tqdm.auto import tqdm

from swebench.harness.apptainer_utils import ensure_apptainer_image, run_apptainer_exec
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_WORKDIR,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_build import close_logger, setup_logger
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import EvaluationError, run_threadpool

APPLY_FAIL_EXIT_CODE = 42
APPTAINER_MOUNT_ROOT = "/swebench-run"
APPTAINER_PATCH = f"{APPTAINER_MOUNT_ROOT}/patch.diff"
APPTAINER_EVAL = f"{APPTAINER_MOUNT_ROOT}/eval.sh"
APPTAINER_TEST_OUTPUT = f"{APPTAINER_MOUNT_ROOT}/test_output.txt"
APPTAINER_DIFF_BEFORE = f"{APPTAINER_MOUNT_ROOT}/git_diff_before.txt"
APPTAINER_DIFF_AFTER = f"{APPTAINER_MOUNT_ROOT}/git_diff_after.txt"
APPTAINER_ORCHESTRATE = f"{APPTAINER_MOUNT_ROOT}/orchestrate.sh"

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def _build_orchestrate_script() -> str:
    lines = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {DOCKER_WORKDIR}",
        "applied=0",
    ]
    for git_apply_cmd in GIT_APPLY_CMDS:
        lines.extend(
            [
                f"if {git_apply_cmd} {APPTAINER_PATCH}; then",
                "  applied=1",
                "  break",
                "fi",
            ]
        )
    lines.extend(
        [
            'if [ "$applied" -ne 1 ]; then',
            f"  echo '{APPLY_PATCH_FAIL}'",
            f"  exit {APPLY_FAIL_EXIT_CODE}",
            "fi",
            f"git -c core.fileMode=false diff > {APPTAINER_DIFF_BEFORE} || true",
            f"/bin/bash {APPTAINER_EVAL} > {APPTAINER_TEST_OUTPUT} 2>&1",
            "eval_rc=$?",
            f"git -c core.fileMode=false diff > {APPTAINER_DIFF_AFTER} || true",
            "exit $eval_rc",
        ]
    )
    return "\n".join(lines) + "\n"


def run_instance_apptainer(
    test_spec: TestSpec,
    pred: dict,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
    apptainer_cache_dir: str | Path = "/tmp/swebench-apptainer",
    apptainer_bin: str = "apptainer",
) -> dict:
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
    report_path = log_dir / LOG_REPORT
    test_output_path = log_dir / LOG_TEST_OUTPUT

    if rewrite_reports:
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return {"completed": True, "resolved": report[instance_id]["resolved"]}

    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {"completed": True, "resolved": report[instance_id]["resolved"]}

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(instance_id, log_dir / LOG_INSTANCE)

    eval_completed = False
    report = {}

    try:
        sif_path = ensure_apptainer_image(
            image_ref=test_spec.instance_image_key,
            cache_dir=apptainer_cache_dir,
            apptainer_bin=apptainer_bin,
        )
        bind_mounts = [(log_dir, APPTAINER_MOUNT_ROOT)]

        patch_file = log_dir / "patch.diff"
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        eval_file = log_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script)
        orchestrate_file = log_dir / "orchestrate.sh"
        orchestrate_file.write_text(_build_orchestrate_script())

        result = run_apptainer_exec(
            sif_path=sif_path,
            inner_cmd=f"/bin/bash {APPTAINER_ORCHESTRATE}",
            bind_mounts=bind_mounts,
            timeout=timeout,
            apptainer_bin=apptainer_bin,
        )
        logger.info(f"Apptainer runtime: {result.runtime:_.2f} seconds")
        logger.info(result.output)

        if result.return_code == APPLY_FAIL_EXIT_CODE:
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{result.output}",
                logger,
            )
        logger.info(APPLY_PATCH_PASS)

        if not test_output_path.exists():
            test_output_path.write_text(result.output)
        if result.timed_out:
            with open(test_output_path, "a") as f:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            raise EvaluationError(
                instance_id,
                f"Test timed out after {timeout} seconds.",
                logger,
            )

        diff_before_path = log_dir / "git_diff_before.txt"
        diff_after_path = log_dir / "git_diff_after.txt"
        if diff_before_path.exists():
            logger.info(f"Git diff before:\n{diff_before_path.read_text().strip()}")
        if diff_after_path.exists():
            logger.info(f"Git diff after:\n{diff_after_path.read_text().strip()}")

        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        eval_completed = True
    except EvaluationError as e:
        logger.info(traceback.format_exc())
        print(e)
    except Exception as e:
        logger.error(
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
    finally:
        close_logger(logger)
        return {
            "completed": eval_completed,
            "resolved": report.get(instance_id, {}).get("resolved", False),
        }


def run_instances_apptainer(
    predictions: dict,
    instances: list,
    max_workers: int,
    run_id: str,
    timeout: int,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    rewrite_reports: bool = False,
    apptainer_cache_dir: str | Path = "/tmp/swebench-apptainer",
    apptainer_bin: str = "apptainer",
):
    test_specs = list(
        map(
            lambda instance: make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
            ),
            instances,
        )
    )

    payloads = []
    for test_spec in test_specs:
        payloads.append(
            (
                test_spec,
                predictions[test_spec.instance_id],
                run_id,
                timeout,
                rewrite_reports,
                apptainer_cache_dir,
                apptainer_bin,
            )
        )

    print(f"Running {len(instances)} instances with Apptainer...")
    stats = {"✓": 0, "✖": 0, "error": 0}
    pbar = tqdm(total=len(payloads), desc="Evaluation", postfix=stats)
    lock = threading.Lock()

    def run_with_progress(*args):
        result = run_instance_apptainer(*args)
        with lock:
            if result["completed"]:
                if result["resolved"]:
                    stats["✓"] += 1
                else:
                    stats["✖"] += 1
            else:
                stats["error"] += 1
            pbar.set_postfix(stats)
            pbar.update()
        return result

    run_threadpool(run_with_progress, payloads, max_workers)
    print("All instances run.")
