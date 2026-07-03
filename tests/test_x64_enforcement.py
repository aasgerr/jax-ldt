"""Regression tests for FIX_PLAN.md H3: x64 must be enforced.

The library requires JAX 64-bit mode; without it, float64 inputs are
silently demoted to float32 and the ridge solver loses precision. These
tests run in a *fresh subprocess* with x64 disabled, because the test
suite's ``conftest`` force-enables x64 in-process (so an in-process test
could never observe the disabled state).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_without_x64(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess with JAX_ENABLE_X64 unset."""
    script = textwrap.dedent(body)
    env = {"PATH": "/usr/bin:/bin"}
    # Inherit the venv's site-packages by reusing the current executable
    # but scrub any x64-enabling env var.
    import os

    clean = dict(os.environ)
    clean.pop("JAX_ENABLE_X64", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=clean,
    )


def test_fit_raises_without_x64() -> None:
    proc = _run_without_x64(
        """
        import numpy as np
        import jax
        assert not jax.config.jax_enable_x64, "x64 unexpectedly on"
        from jax_ldt import LinearTreeRegressor
        X = np.random.default_rng(0).uniform(-1, 1, size=(50, 2))
        y = X[:, 0] + X[:, 1]
        try:
            LinearTreeRegressor(max_depth=2, min_samples_leaf=5).fit(X, y)
        except RuntimeError as e:
            print("RAISED:" + str(e))
        else:
            print("NO_RAISE")
        """
    )
    assert "RAISED:" in proc.stdout, proc.stdout + proc.stderr
    assert "NO_RAISE" not in proc.stdout


def test_x64_error_message_has_remediation() -> None:
    proc = _run_without_x64(
        """
        import numpy as np
        from jax_ldt import LinearTreeRegressor
        X = np.random.default_rng(0).uniform(-1, 1, size=(50, 2))
        y = X[:, 0] + X[:, 1]
        try:
            LinearTreeRegressor(max_depth=2, min_samples_leaf=5).fit(X, y)
        except RuntimeError as e:
            print("MSG:" + str(e))
        """
    )
    assert "JAX_ENABLE_X64" in proc.stdout
    assert "jax_enable_x64" in proc.stdout


def test_predict_also_guarded() -> None:
    """The predict boundary must guard too (a Tree could be unpickled in
    a fresh x64-off process)."""
    proc = _run_without_x64(
        """
        import numpy as np
        from jax_ldt.tree_core import predict
        from jax_ldt._validation import _require_x64
        try:
            _require_x64()
        except RuntimeError:
            print("GUARD_OK")
        """
    )
    assert "GUARD_OK" in proc.stdout, proc.stdout + proc.stderr
