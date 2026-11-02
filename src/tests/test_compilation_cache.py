"""Tests for the opt-in jax compilation cache."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

import sax
import sax.utils


def test_sets_the_jax_config(tmp_path: Path) -> None:
    directory = sax.enable_compilation_cache(tmp_path / "cache")

    assert Path(jax.config.jax_compilation_cache_dir) == tmp_path / "cache"
    assert directory == tmp_path / "cache"
    assert directory.is_dir()
    # the whole point of the feature on sax's compilation sizes: jax's default of 1s
    # would skip every one of them
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0


def test_a_min_compile_time_can_be_raised(tmp_path: Path) -> None:
    sax.enable_compilation_cache(tmp_path / "cache", min_compile_time_secs=2.5)

    assert jax.config.jax_persistent_cache_min_compile_time_secs == 2.5


def test_honours_the_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAX_CACHE_DIR", str(tmp_path / "from-env"))

    assert sax.enable_compilation_cache() == tmp_path / "from-env"


def test_an_explicit_path_beats_the_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SAX_CACHE_DIR", str(tmp_path / "from-env"))

    assert sax.enable_compilation_cache(tmp_path / "explicit") == tmp_path / "explicit"


def test_default_is_scoped_to_the_machine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sax.utils, "_cpu_features", lambda: "flags-one,flags-two")
    first = sax.utils._default_cache_dir()  # noqa: SLF001

    monkeypatch.setattr(sax.utils, "_cpu_features", lambda: "flags-three")
    second = sax.utils._default_cache_dir()  # noqa: SLF001

    assert first.parent == second.parent == tmp_path / "sax"
    # a host whose cpu features differ must not read this machine's binaries...
    assert first != second
    # ...while the same machine keeps landing in the same place
    assert sax.utils._default_cache_dir() == second  # noqa: SLF001


def test_a_second_process_reuses_the_cache(tmp_path: Path) -> None:
    """The point of the cache: process one compiles and writes, process two reuses."""
    cache = tmp_path / "jax"
    script = (
        "import jax, jax.numpy as jnp\n"
        "import sax\n"
        f"sax.enable_compilation_cache({str(cache)!r})\n"
        "print(float(jax.jit(lambda x: (x * 2).sum())(jnp.arange(10.0))))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

    first = run()
    entries = sorted(p for p in cache.rglob("*") if p.is_file())
    assert entries, "the first process wrote no cache entries"

    second = run()
    assert second.stdout == first.stdout
    assert sorted(p for p in cache.rglob("*") if p.is_file()) == entries, (
        "the second process recompiled instead of reusing the cache"
    )
