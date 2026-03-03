from __future__ import annotations

import shutil
import warnings


def _validate_cap_add_support(test_specs: list) -> None:
    """
    Raise when a test spec requests Docker-only capability additions.
    """
    unsupported = []
    for spec in test_specs:
        run_args = spec.docker_specs.get("run_args", {})
        cap_add = run_args.get("cap_add", [])
        if cap_add:
            unsupported.append((spec.instance_id, cap_add))

    if unsupported:
        unsupported_str = ", ".join(
            f"{instance_id} (cap_add={cap_add})"
            for instance_id, cap_add in unsupported[:10]
        )
        if len(unsupported) > 10:
            unsupported_str += f", ... (+{len(unsupported) - 10} more)"
        raise ValueError(
            "Apptainer eval-only mode does not support Docker cap_add requests. "
            f"Unsupported instances: {unsupported_str}"
        )


def validate_apptainer_runtime_config(
    namespace: str | None,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    test_specs: list,
) -> None:
    """
    Validate runtime configuration for eval-only Apptainer mode.
    """
    if shutil.which("apptainer") is None:
        raise RuntimeError(
            "Apptainer runtime requested but `apptainer` was not found on PATH."
        )

    if namespace is None:
        raise ValueError(
            "Apptainer eval-only mode requires a namespace for remote images. "
            "Do not use --namespace none."
        )

    if force_rebuild:
        raise ValueError(
            "--force_rebuild is not supported for --runtime apptainer in eval-only mode."
        )

    if clean:
        warnings.warn(
            "--clean is ignored for --runtime apptainer in eval-only mode.",
            UserWarning,
        )

    if cache_level != "env":
        warnings.warn(
            "--cache_level is ignored for --runtime apptainer in eval-only mode.",
            UserWarning,
        )

    _validate_cap_add_support(test_specs)
