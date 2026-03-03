import warnings

from types import SimpleNamespace

import pytest

from swebench.harness.runtime_checks import validate_apptainer_runtime_config


def _spec(instance_id: str, cap_add: list[str] | None = None):
    return SimpleNamespace(
        instance_id=instance_id,
        docker_specs={"run_args": {"cap_add": cap_add or []}},
    )


def test_validate_apptainer_runtime_config_requires_binary(monkeypatch):
    monkeypatch.setattr("swebench.harness.runtime_checks.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="`apptainer` was not found on PATH"):
        validate_apptainer_runtime_config(
            namespace="swebench",
            force_rebuild=False,
            cache_level="env",
            clean=False,
            test_specs=[],
        )


def test_validate_apptainer_runtime_config_requires_namespace(monkeypatch):
    monkeypatch.setattr(
        "swebench.harness.runtime_checks.shutil.which",
        lambda _: "/usr/bin/apptainer",
    )

    with pytest.raises(ValueError, match="Do not use --namespace none"):
        validate_apptainer_runtime_config(
            namespace=None,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            test_specs=[],
        )


def test_validate_apptainer_runtime_config_rejects_force_rebuild(monkeypatch):
    monkeypatch.setattr(
        "swebench.harness.runtime_checks.shutil.which",
        lambda _: "/usr/bin/apptainer",
    )

    with pytest.raises(ValueError, match="--force_rebuild is not supported"):
        validate_apptainer_runtime_config(
            namespace="swebench",
            force_rebuild=True,
            cache_level="env",
            clean=False,
            test_specs=[],
        )


def test_validate_apptainer_runtime_config_warns_for_ignored_flags(monkeypatch):
    monkeypatch.setattr(
        "swebench.harness.runtime_checks.shutil.which",
        lambda _: "/usr/bin/apptainer",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_apptainer_runtime_config(
            namespace="swebench",
            force_rebuild=False,
            cache_level="instance",
            clean=True,
            test_specs=[],
        )

    messages = [str(w.message) for w in caught]
    assert any("--clean is ignored" in msg for msg in messages)
    assert any("--cache_level is ignored" in msg for msg in messages)


def test_validate_apptainer_runtime_config_rejects_cap_add(monkeypatch):
    monkeypatch.setattr(
        "swebench.harness.runtime_checks.shutil.which",
        lambda _: "/usr/bin/apptainer",
    )
    specs = [
        _spec("sympy__sympy-20590"),
        _spec("chartjs__chartjs-1234", cap_add=["SYS_ADMIN"]),
    ]

    with pytest.raises(
        ValueError, match="does not support Docker cap_add requests"
    ) as exc:
        validate_apptainer_runtime_config(
            namespace="swebench",
            force_rebuild=False,
            cache_level="env",
            clean=False,
            test_specs=specs,
        )

    assert "chartjs__chartjs-1234" in str(exc.value)
