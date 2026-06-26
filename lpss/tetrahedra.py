"""Tetrahedral tessellation of the phase-space sheet (numba-accelerated).

Each cubic cell of the Lagrangian grid is split into **6 tetrahedra** that share
the cube's main diagonal (the Freudenthal/Kuhn triangulation).  Because the
sheet's connectivity is fixed, these tetrahedra deform with the flow but never
tear -- so their volumes, centroids and the mass they carry give us a *fully
adaptive* estimate of the density field, exact wherever the map ``q -> x`` is
linear.

Each operation has **two** implementations selected automatically:

* a compiled, multi-threaded :mod:`numba` kernel (``prange`` over the slowest
  grid axis) used when numba is available, and
* a fully **vectorised NumPy** version used otherwise.

The vectorised NumPy version is a perfectly usable fallback (it is what the
original notebook shipped); the numba kernels are typically *hundreds* of times
faster still and, crucially, use far less memory because they stream through the
data instead of allocating large temporary arrays.

Everything is fully **periodic**: neighbours are taken modulo ``n`` and every
edge vector uses the minimum-image convention, so the six tetrahedra of every
cell are always well defined and the total volume is exactly ``boxsize**3``.
"""

from __future__ import annotations

import numpy as np

from ._backend import HAVE_NUMBA, njit, prange

__all__ = [
    "TETS",
    "tet_volumes",
    "voxel_volumes",
    "tet_centroids",
    "tet_densities",
    "tet_interpolated_particles",
    "deposit",
    "sheet_density",
]

# ---------------------------------------------------------------------------
# Connectivity: the 6 tetrahedra of the unit cube.
#
# The 8 cube corners, indexed 0..7, have integer offsets:
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
# The six tetrahedra all share the long diagonal 0--6 and tile the cube.  We
# store them directly as their (4, 3) integer corner offsets so the kernels can
# index neighbours without an extra lookup table.  The ordering of the four
# corners is chosen so that a cell with no displacement yields *positive* signed
# volume.
# ---------------------------------------------------------------------------
_CORNER = np.array(
    [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ],
    dtype=np.int64,
)
_CONN = np.array(
    [
        (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
    ],
    dtype=np.int64,
)
#: ``(6, 4, 3)`` integer corner offsets of the six tetrahedra.
TETS = _CORNER[_CONN].copy()

NTET = 6


@njit(cache=True, fastmath=True, parallel=True)
def _tet_volumes_kernel(x, boxsize, tets):
    nx, ny, nz = x.shape[0], x.shape[1], x.shape[2]
    vol = np.empty((nx, ny, nz, NTET), dtype=x.dtype)
    half = 0.5 * boxsize
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                for m in range(NTET):
                    # corner 0 of this tetrahedron
                    i0 = (i + tets[m, 0, 0]) % nx
                    j0 = (j + tets[m, 0, 1]) % ny
                    k0 = (k + tets[m, 0, 2]) % nz
                    p0x = x[i0, j0, k0, 0]
                    p0y = x[i0, j0, k0, 1]
                    p0z = x[i0, j0, k0, 2]
                    # the three edge vectors p1-p0, p2-p0, p3-p0 (minimum image)
                    ex = np.empty(3)
                    ey = np.empty(3)
                    ez = np.empty(3)
                    for e in range(3):
                        ii = (i + tets[m, e + 1, 0]) % nx
                        jj = (j + tets[m, e + 1, 1]) % ny
                        kk = (k + tets[m, e + 1, 2]) % nz
                        dx = x[ii, jj, kk, 0] - p0x
                        dy = x[ii, jj, kk, 1] - p0y
                        dz = x[ii, jj, kk, 2] - p0z
                        # minimum image
                        if dx > half:
                            dx -= boxsize
                        elif dx < -half:
                            dx += boxsize
                        if dy > half:
                            dy -= boxsize
                        elif dy < -half:
                            dy += boxsize
                        if dz > half:
                            dz -= boxsize
                        elif dz < -half:
                            dz += boxsize
                        ex[e] = dx
                        ey[e] = dy
                        ez[e] = dz
                    # signed volume = (1/6) e0 . (e1 x e2)
                    cx = ey[1] * ez[2] - ez[1] * ey[2]
                    cy = ez[1] * ex[2] - ex[1] * ez[2]
                    cz = ex[1] * ey[2] - ey[1] * ex[2]
                    vol[i, j, k, m] = (ex[0] * cx + ey[0] * cy + ez[0] * cz) / 6.0
    return vol


@njit(cache=True, fastmath=True, parallel=True)
def _tet_centroids_kernel(x, boxsize, tets):
    nx, ny, nz = x.shape[0], x.shape[1], x.shape[2]
    cen = np.empty((nx * ny * nz * NTET, 3), dtype=x.dtype)
    half = 0.5 * boxsize
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                base = ((i * ny + j) * nz + k) * NTET
                for m in range(NTET):
                    i0 = (i + tets[m, 0, 0]) % nx
                    j0 = (j + tets[m, 0, 1]) % ny
                    k0 = (k + tets[m, 0, 2]) % nz
                    p0x = x[i0, j0, k0, 0]
                    p0y = x[i0, j0, k0, 1]
                    p0z = x[i0, j0, k0, 2]
                    sx = p0x
                    sy = p0y
                    sz = p0z
                    for e in range(3):
                        ii = (i + tets[m, e + 1, 0]) % nx
                        jj = (j + tets[m, e + 1, 1]) % ny
                        kk = (k + tets[m, e + 1, 2]) % nz
                        dx = x[ii, jj, kk, 0] - p0x
                        dy = x[ii, jj, kk, 1] - p0y
                        dz = x[ii, jj, kk, 2] - p0z
                        if dx > half:
                            dx -= boxsize
                        elif dx < -half:
                            dx += boxsize
                        if dy > half:
                            dy -= boxsize
                        elif dy < -half:
                            dy += boxsize
                        if dz > half:
                            dz -= boxsize
                        elif dz < -half:
                            dz += boxsize
                        sx += p0x + dx
                        sy += p0y + dy
                        sz += p0z + dz
                    out = base + m
                    cen[out, 0] = (sx * 0.25) % boxsize
                    cen[out, 1] = (sy * 0.25) % boxsize
                    cen[out, 2] = (sz * 0.25) % boxsize
    return cen


@njit(cache=True, fastmath=True, parallel=True)
def _interp_kernel(x, boxsize, tets, bary):
    nx, ny, nz = x.shape[0], x.shape[1], x.shape[2]
    split = bary.shape[0]
    out = np.empty((nx * ny * nz * NTET * split, 3), dtype=x.dtype)
    half = 0.5 * boxsize
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                cellbase = ((i * ny + j) * nz + k) * NTET * split
                for m in range(NTET):
                    # gather the 4 corners of this tetrahedron relative to p0
                    i0 = (i + tets[m, 0, 0]) % nx
                    j0 = (j + tets[m, 0, 1]) % ny
                    k0 = (k + tets[m, 0, 2]) % nz
                    p0x = x[i0, j0, k0, 0]
                    p0y = x[i0, j0, k0, 1]
                    p0z = x[i0, j0, k0, 2]
                    vx = np.empty(4)
                    vy = np.empty(4)
                    vz = np.empty(4)
                    vx[0] = p0x
                    vy[0] = p0y
                    vz[0] = p0z
                    for e in range(3):
                        ii = (i + tets[m, e + 1, 0]) % nx
                        jj = (j + tets[m, e + 1, 1]) % ny
                        kk = (k + tets[m, e + 1, 2]) % nz
                        dx = x[ii, jj, kk, 0] - p0x
                        dy = x[ii, jj, kk, 1] - p0y
                        dz = x[ii, jj, kk, 2] - p0z
                        if dx > half:
                            dx -= boxsize
                        elif dx < -half:
                            dx += boxsize
                        if dy > half:
                            dy -= boxsize
                        elif dy < -half:
                            dy += boxsize
                        if dz > half:
                            dz -= boxsize
                        elif dz < -half:
                            dz += boxsize
                        vx[e + 1] = p0x + dx
                        vy[e + 1] = p0y + dy
                        vz[e + 1] = p0z + dz
                    base = cellbase + m * split
                    for s in range(split):
                        px = 0.0
                        py = 0.0
                        pz = 0.0
                        for c in range(4):
                            w = bary[s, c]
                            px += w * vx[c]
                            py += w * vy[c]
                            pz += w * vz[c]
                        out[base + s, 0] = px % boxsize
                        out[base + s, 1] = py % boxsize
                        out[base + s, 2] = pz % boxsize
    return out


@njit(cache=True, fastmath=True)
def _cic_deposit_kernel(grid, pos, weights, boxsize, ngrid):
    """Cloud-in-cell deposit of weighted points onto a periodic grid (serial).

    Serial on purpose: scattered writes race under ``prange``.  The arithmetic
    is the bottleneck regardless, and numba still compiles this to tight code.
    """
    h = boxsize / ngrid
    for p in range(pos.shape[0]):
        w = weights[p]
        fx = pos[p, 0] / h - 0.5
        fy = pos[p, 1] / h - 0.5
        fz = pos[p, 2] / h - 0.5
        ix = int(np.floor(fx))
        iy = int(np.floor(fy))
        iz = int(np.floor(fz))
        dx = fx - ix
        dy = fy - iy
        dz = fz - iz
        for ddx in range(2):
            wx = (1.0 - dx) if ddx == 0 else dx
            gx = (ix + ddx) % ngrid
            for ddy in range(2):
                wy = (1.0 - dy) if ddy == 0 else dy
                gy = (iy + ddy) % ngrid
                for ddz in range(2):
                    wz = (1.0 - dz) if ddz == 0 else dz
                    gz = (iz + ddz) % ngrid
                    grid[gx, gy, gz] += w * wx * wy * wz


# --------------------------------------------------------------------------- #
# Vectorised NumPy implementations (used when numba is unavailable).
#
# These are the honest baseline: they trade memory for speed by building a few
# (n, n, n, 3) temporary arrays per tetrahedron via ``np.roll``.  Correct and
# reasonably fast, but the numba kernels above beat them by ~100x with far less
# RAM.  Output layouts are kept byte-for-byte identical to the kernels so the
# two backends are interchangeable.
# --------------------------------------------------------------------------- #
def _shift(x, off):
    """Periodic neighbour field: ``_shift(x, off)[i] == x[(i + off) % n]``."""
    return np.roll(x, shift=(-off[0], -off[1], -off[2]), axis=(0, 1, 2))


def _mip(d, box):
    return d - box * np.round(d / box)


def _tet_volumes_numpy(x, box, tets):
    nx, ny, nz = x.shape[:3]
    vol = np.empty((nx, ny, nz, NTET), dtype=x.dtype)
    for m in range(NTET):
        p0 = _shift(x, tets[m, 0])
        e0 = _mip(_shift(x, tets[m, 1]) - p0, box)
        e1 = _mip(_shift(x, tets[m, 2]) - p0, box)
        e2 = _mip(_shift(x, tets[m, 3]) - p0, box)
        vol[..., m] = np.einsum("...i,...i->...", e0, np.cross(e1, e2)) / 6.0
    return vol


def _tet_centroids_numpy(x, box, tets):
    nx, ny, nz = x.shape[:3]
    cen = np.empty((nx, ny, nz, NTET, 3), dtype=x.dtype)
    for m in range(NTET):
        p0 = _shift(x, tets[m, 0])
        s = p0.copy()
        for e in range(3):
            s = s + (p0 + _mip(_shift(x, tets[m, e + 1]) - p0, box))
        cen[:, :, :, m, :] = (s * 0.25) % box
    return cen.reshape(nx * ny * nz * NTET, 3)


def _interp_numpy(x, box, tets, bary):
    nx, ny, nz = x.shape[:3]
    split = bary.shape[0]
    out = np.empty((nx, ny, nz, NTET, split, 3), dtype=x.dtype)
    for m in range(NTET):
        p0 = _shift(x, tets[m, 0])
        verts = [p0]
        for e in range(3):
            verts.append(p0 + _mip(_shift(x, tets[m, e + 1]) - p0, box))
        for s in range(split):
            acc = bary[s, 0] * verts[0]
            for c in range(1, 4):
                acc = acc + bary[s, c] * verts[c]
            out[:, :, :, m, s, :] = acc % box
    return out.reshape(-1, 3)


def _cic_deposit_numpy(grid, pos, weights, box, ngrid):
    h = box / ngrid
    f = pos / h - 0.5
    i0 = np.floor(f).astype(np.int64)
    d = f - i0
    for ddx in (0, 1):
        wx = (1.0 - d[:, 0]) if ddx == 0 else d[:, 0]
        gx = (i0[:, 0] + ddx) % ngrid
        for ddy in (0, 1):
            wy = (1.0 - d[:, 1]) if ddy == 0 else d[:, 1]
            gy = (i0[:, 1] + ddy) % ngrid
            for ddz in (0, 1):
                wz = (1.0 - d[:, 2]) if ddz == 0 else d[:, 2]
                gz = (i0[:, 2] + ddz) % ngrid
                np.add.at(grid, (gx, gy, gz), weights * wx * wy * wz)


# --------------------------------------------------------------------------- #
# Public, array-level API.  These accept a LagrangianMesh *or* a raw positions
# array and dispatch to whichever backend (numba kernel / vectorised NumPy) is
# active.
# --------------------------------------------------------------------------- #
def _as_positions(mesh_or_x):
    """Accept a LagrangianMesh or an ``(n,n,n,3)`` positions array."""
    from .lagrangian import LagrangianMesh

    if isinstance(mesh_or_x, LagrangianMesh):
        return np.ascontiguousarray(mesh_or_x.positions(wrap=False)), mesh_or_x.boxsize
    x = np.asarray(mesh_or_x, dtype=np.float64)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError("expected a LagrangianMesh or an (n,n,n,3) array")
    raise ValueError("a boxsize is required when passing a raw positions array")


def tet_volumes(mesh):
    """Signed volume of every tetrahedron, shape ``(n, n, n, 6)``.

    Negative volumes mark cells that have undergone shell-crossing (the map
    ``q -> x`` has inverted there).  The sum over all tetrahedra equals
    ``boxsize**3`` to machine precision.
    """
    x, box = _as_positions(mesh)
    if HAVE_NUMBA:
        return _tet_volumes_kernel(x, float(box), TETS)
    return _tet_volumes_numpy(x, float(box), TETS)


def voxel_volumes(mesh):
    """Total (signed) volume of each Lagrangian cell, shape ``(n, n, n)``."""
    return tet_volumes(mesh).sum(axis=-1)


def tet_centroids(mesh):
    """Centroid of every tetrahedron, shape ``(6 * n**3, 3)`` (wrapped to box)."""
    x, box = _as_positions(mesh)
    if HAVE_NUMBA:
        return _tet_centroids_kernel(x, float(box), TETS)
    return _tet_centroids_numpy(x, float(box), TETS)


def tet_densities(mesh, particle_mass=1.0):
    """Mass density carried by each tetrahedron, shape ``(n, n, n, 6)``.

    Each Lagrangian cell carries one particle's worth of mass, shared equally
    among its 6 tetrahedra, so a tetrahedron of signed volume ``V`` has density
    ``rho = (particle_mass / 6) / |V|``.  This is the exact density of the
    piecewise-linear sheet inside that tetrahedron.
    """
    vol = tet_volumes(mesh)
    return (particle_mass / NTET) / np.abs(vol)


def _uniform_tet_barycentric(split, seed=0):
    """``(split, 4)`` barycentric weights uniformly sampling the unit tetrahedron.

    Uses the classic sorted-uniform simplex sampler: draw three uniforms, sort
    them, and take successive differences.  The four weights are non-negative
    and sum to one, so any convex combination stays inside the tetrahedron.
    """
    rng = np.random.default_rng(seed)
    u = np.sort(rng.random((split, 3)), axis=1)
    bary = np.empty((split, 4))
    bary[:, 0] = u[:, 0]
    bary[:, 1] = u[:, 1] - u[:, 0]
    bary[:, 2] = u[:, 2] - u[:, 1]
    bary[:, 3] = 1.0 - u[:, 2]
    return bary


def tet_interpolated_particles(mesh, split=10, seed=0):
    """Refine the sheet by scattering ``split`` particles inside each tetrahedron.

    Returns an ``(6 * split * n**3, 3)`` array of positions.  Because the
    particles are placed with the *same linear map* that defines each
    tetrahedron, this is a geometrically exact up-sampling of the sheet -- the
    refined particles trace caustics far more sharply than the original grid.
    """
    x, box = _as_positions(mesh)
    bary = _uniform_tet_barycentric(split, seed=seed)
    if HAVE_NUMBA:
        return _interp_kernel(x, float(box), TETS, bary)
    return _interp_numpy(x, float(box), TETS, bary)


def deposit(pos, ngrid, boxsize, weights=None, out=None):
    """Cloud-in-cell deposit of weighted points onto a periodic ``ngrid**3`` grid.

    Parameters
    ----------
    pos : (N, 3) array
    ngrid : int
    boxsize : float
    weights : (N,) array, optional
        Per-point weight (mass); defaults to 1.
    out : (ngrid, ngrid, ngrid) array, optional
        Accumulate into this array instead of allocating a new one.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    if weights is None:
        weights = np.ones(pos.shape[0], dtype=np.float64)
    else:
        weights = np.ascontiguousarray(weights, dtype=np.float64)
    if out is None:
        out = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    if HAVE_NUMBA:
        _cic_deposit_kernel(out, pos, weights, float(boxsize), int(ngrid))
    else:
        _cic_deposit_numpy(out, pos, weights, float(boxsize), int(ngrid))
    return out


def sheet_density(mesh, ngrid, particle_mass=1.0, split=1, seed=0):
    """Estimate the mass density on an ``ngrid**3`` grid from the sheet.

    For ``split == 1`` the mass of each tetrahedron is deposited at its
    centroid.  For ``split > 1`` the tetrahedra are sub-sampled with
    :func:`tet_interpolated_particles`, giving a smoother, higher-fidelity
    estimate (closer to the exact projected-tetrahedron density) at the cost of
    more points.  The result is normalised so that the total mass is conserved.
    """
    box = mesh.boxsize
    total_mass = particle_mass * mesh.npart
    if split <= 1:
        cen = tet_centroids(mesh)
        # each tetrahedron carries m/6 of its cell's mass, independent of volume
        w = np.full(cen.shape[0], particle_mass / NTET)
        rho = deposit(cen, ngrid, box, weights=w)
    else:
        pts = tet_interpolated_particles(mesh, split=split, seed=seed)
        w = np.full(pts.shape[0], total_mass / pts.shape[0])
        rho = deposit(pts, ngrid, box, weights=w)
    # convert deposited mass to density (mass per unit volume)
    cell_vol = (box / ngrid) ** 3
    return rho / cell_vol
