"""Correctness tests for the :mod:`lpss` library.

Run with ``pytest``.  These cover the properties that *must* hold regardless of
backend: volume conservation, Fourier exactness, mass conservation, and that the
numba and pure-NumPy paths agree.
"""

import os

import numpy as np
import pytest

import lpss
from lpss import tetrahedra as tet
from lpss.fourier import fourier_upsample


# --------------------------------------------------------------------------- #
# Tetrahedral tessellation
# --------------------------------------------------------------------------- #
def test_total_volume_conserved():
    """The six tetrahedra of every cell tile space: sum == boxsize**3 exactly."""
    mesh = lpss.LagrangianMesh.zeldovich(n=24, boxsize=70.0, growth=1.3, seed=5)
    vol = tet.tet_volumes(mesh)
    assert np.isclose(vol.sum(), mesh.boxsize ** 3, rtol=1e-10)


def test_uniform_grid_has_unit_cells():
    """With zero displacement every cell has volume == cell**3 and is positive."""
    mesh = lpss.LagrangianMesh(
        disp=np.zeros((16, 16, 16, 3)), boxsize=16.0
    )
    vol = tet.voxel_volumes(mesh)
    assert np.allclose(vol, 1.0)  # cell size is 1.0
    assert (tet.tet_volumes(mesh) > 0).all()


def test_shell_crossing_creates_negative_volumes():
    """Cranking up the growth factor must invert some tetrahedra."""
    calm = lpss.LagrangianMesh.zeldovich(n=24, boxsize=70.0, growth=0.1, seed=5)
    wild = lpss.LagrangianMesh.zeldovich(n=24, boxsize=70.0, growth=3.0, seed=5)
    assert (tet.tet_volumes(wild) < 0).mean() > (tet.tet_volumes(calm) < 0).mean()


def test_interpolated_particles_inside_box_and_count():
    mesh = lpss.LagrangianMesh.zeldovich(n=12, boxsize=40.0, growth=0.7, seed=2)
    pts = tet.tet_interpolated_particles(mesh, split=8, seed=0)
    assert pts.shape == (6 * 8 * 12 ** 3, 3)
    assert pts.min() >= 0.0 and pts.max() < mesh.boxsize + 1e-9


def test_sheet_density_conserves_mass():
    mesh = lpss.LagrangianMesh.zeldovich(n=16, boxsize=50.0, growth=0.8, seed=4)
    for split in (1, 4):
        rho = tet.sheet_density(mesh, ngrid=16, particle_mass=1.0, split=split)
        cell_vol = (mesh.boxsize / 16) ** 3
        assert np.isclose((rho * cell_vol).sum(), mesh.n ** 3, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Fourier up-sampling
# --------------------------------------------------------------------------- #
def _bandlimited(X, Y, Z, L):
    k = 2 * np.pi / L
    return np.sin(2 * k * X) + 0.5 * np.cos(3 * k * Y) * np.sin(k * Z)


@pytest.mark.parametrize("n", [8, 9, 16])
@pytest.mark.parametrize("factor", [2, 3])
def test_fourier_upsample_is_exact(n, factor):
    """Spectral up-sampling reproduces a band-limited field to machine precision."""
    L = 1.0
    x = L * np.arange(n) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    f = _bandlimited(X, Y, Z, L)
    up = fourier_upsample(f, [factor, factor, factor])
    m = n * factor
    xm = L * np.arange(m) / m
    Xm, Ym, Zm = np.meshgrid(xm, xm, xm, indexing="ij")
    assert np.allclose(up, _bandlimited(Xm, Ym, Zm, L), atol=1e-12)


def test_upsample_mesh_reproduces_coarse_nodes():
    """The coarse grid values survive untouched in the up-sampled mesh."""
    mesh = lpss.LagrangianMesh.zeldovich(n=16, boxsize=50.0, growth=0.5, seed=3)
    fine = lpss.upsample_mesh(mesh, 2)
    assert fine.n == 32
    assert np.allclose(fine.disp[::2, ::2, ::2], mesh.disp, atol=1e-9)
    assert fine.vel is not None
    assert np.allclose(fine.vel[::2, ::2, ::2], mesh.vel, atol=1e-9)


def test_upsample_mesh_preserves_volume():
    mesh = lpss.LagrangianMesh.zeldovich(n=16, boxsize=50.0, growth=0.6, seed=7)
    fine = lpss.upsample_mesh(mesh, 2)
    vol = tet.tet_volumes(fine)
    assert np.isclose(vol.sum(), fine.boxsize ** 3, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Backend agreement: numba result == pure-NumPy result
# --------------------------------------------------------------------------- #
def test_numba_matches_numpy_subprocess():
    """Run the same computation with numba disabled and check identical output."""
    import subprocess
    import sys

    script = (
        "import numpy as np, lpss;"
        "from lpss import tetrahedra as tet;"
        "m=lpss.LagrangianMesh.zeldovich(n=16,boxsize=50.0,growth=1.0,seed=11);"
        "v=tet.tet_volumes(m);"
        "print(repr((float(v.sum()), float(np.abs(v).max()), int((v<0).sum()))))"
    )
    env = dict(os.environ)

    env["LPSS_DISABLE_NUMBA"] = "0"
    out_numba = subprocess.check_output([sys.executable, "-c", script], env=env)
    env["LPSS_DISABLE_NUMBA"] = "1"
    out_numpy = subprocess.check_output([sys.executable, "-c", script], env=env)

    a = eval(out_numba)
    b = eval(out_numpy)
    assert np.allclose(a, b, rtol=1e-10), (a, b)


# --------------------------------------------------------------------------- #
# Gadget reader (only if the sample snapshot is present)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "snapshot_010")),
    reason="sample snapshot not available",
)
def test_gadget_roundtrip():
    snap = lpss.read_gadget(
        os.path.join(os.path.dirname(__file__), "..", "snapshot_010")
    )
    assert snap.pos.shape == (262144, 3)
    assert np.isclose(snap.boxsize, 40.0)
    mesh = lpss.LagrangianMesh.from_gadget(snap)
    vol = tet.tet_volumes(mesh)
    # voids (positive volume) dominate the total volume budget
    assert vol.sum() > 0
    assert np.isclose(vol.sum(), mesh.boxsize ** 3, rtol=1e-6)
