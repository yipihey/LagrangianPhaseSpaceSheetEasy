"""Benchmark the numba backend against the pure-NumPy fallback.

Run directly::

    python benchmark.py            # uses whatever backend is active
    LPSS_DISABLE_NUMBA=1 python benchmark.py   # force the NumPy path

To compare the two it is easiest to run the helper which launches both in
subprocesses (so the backend choice is picked up cleanly at import time)::

    python benchmark.py --compare
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def _time(func, *args, repeat=3, warmup=1, **kwargs):
    for _ in range(warmup):  # trigger numba JIT compilation, fill caches
        func(*args, **kwargs)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def run(n=64, box=100.0, growth=1.5):
    import lpss
    from lpss import tetrahedra as tet

    mesh = lpss.LagrangianMesh.zeldovich(n=n, boxsize=box, growth=growth, seed=0)
    print(f"backend: {lpss.numba_info()}")
    print(f"grid: {n}^3 = {n**3:,} particles, {6*n**3:,} tetrahedra\n")

    results = {}
    results["tet_volumes"] = _time(tet.tet_volumes, mesh)
    results["tet_centroids"] = _time(tet.tet_centroids, mesh)
    results["interp x8/tet"] = _time(tet.tet_interpolated_particles, mesh, split=8)
    results["sheet_density"] = _time(tet.sheet_density, mesh, n)
    results["fourier x2"] = _time(lpss.upsample_mesh, mesh, 2)

    for name, t in results.items():
        print(f"  {name:18s}: {t*1e3:8.1f} ms")
    return results


def compare(n=64):
    import subprocess

    script = (
        "import benchmark, json, sys;"
        f"r=benchmark.run(n={n});"
        "print('JSON'+json.dumps(r))"
    )
    out = {}
    for disable in ("0", "1"):
        env = dict(os.environ, LPSS_DISABLE_NUMBA=disable)
        print("=" * 60)
        proc = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            continue
        import json

        line = [l for l in proc.stdout.splitlines() if l.startswith("JSON")][0]
        out["numba" if disable == "0" else "numpy"] = json.loads(line[4:])

    if "numba" in out and "numpy" in out:
        print("=" * 60)
        print(f"\n{'operation':18s} {'numpy (ms)':>12s} {'numba (ms)':>12s} {'speedup':>9s}")
        for key in out["numba"]:
            a = out["numpy"][key] * 1e3
            b = out["numba"][key] * 1e3
            print(f"{key:18s} {a:12.1f} {b:12.1f} {a/b:8.1f}x")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    else:
        run()
