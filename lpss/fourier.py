"""Spectral (Fourier) up-sampling of the Lagrangian sheet.

The displacement field ``Psi(q)`` and the velocity field ``v(q)`` are smooth,
*periodic* functions sampled on the regular Lagrangian grid.  A periodic
band-limited function is fully determined by its samples, so we can evaluate it
at **any** finer regular grid *exactly* by zero-padding its discrete Fourier
transform -- this is sinc / spectral interpolation, the most accurate
interpolation possible for such a field.

This is the right tool for "I have an ``n**3`` Lagrangian mesh and I want an
``(f*n)**3`` one": refine ``Psi`` and ``v`` spectrally, lay down the finer grid
of ``q`` values, and read off up-sampled positions ``x = q + Psi`` and
velocities ``v``.  Unlike scattering random particles inside tetrahedra, the
result is a *structured* finer mesh that itself tessellates -- so it can be fed
straight back into the tetrahedron machinery.

Why up-sample the displacement and not the positions?  Because the positions
wrap around the periodic box and are therefore *not* smooth functions of ``q``
once particles have crossed the boundary, while ``Psi`` always is.

The only subtlety is the Nyquist mode of an even-sized axis: it must be split
symmetrically between ``+k_Ny`` and ``-k_Ny`` in the padded spectrum so the
up-sampled field stays real and centred.  :func:`fourier_upsample` handles this
per axis.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fourier_upsample", "upsample_mesh"]


def _normalize_factors(factor, ndim):
    if np.isscalar(factor):
        factors = [int(factor)] * ndim
    else:
        factors = [int(f) for f in factor]
    if len(factors) != ndim:
        raise ValueError(f"expected {ndim} integer factors, got {factor!r}")
    if any(f < 1 for f in factors):
        raise ValueError("upsampling factors must be >= 1")
    return factors


def _pad_axis(spec, axis, factor):
    """Zero-pad the (already FFT'd) spectrum along one axis by ``factor``.

    ``spec`` is the full complex FFT along ``axis``.  We grow that axis from
    ``n`` to ``m = factor * n`` frequency slots, copying the low positive and
    high negative frequencies into place and splitting the Nyquist mode of an
    even-length axis so the inverse transform is real.
    """
    n = spec.shape[axis]
    if factor == 1:
        return spec
    m = n * factor

    newshape = list(spec.shape)
    newshape[axis] = m
    out = np.zeros(newshape, dtype=complex)

    def sl(a, b):
        idx = [slice(None)] * spec.ndim
        idx[axis] = slice(a, b)
        return tuple(idx)

    if n % 2 == 1:
        h = (n + 1) // 2  # number of non-negative frequencies (incl. DC)
        out[sl(0, h)] = spec[sl(0, h)]          # 0 .. +k_max
        out[sl(m - (n - h), m)] = spec[sl(h, n)]  # negative frequencies
    else:
        h = n // 2  # index h is the Nyquist mode
        out[sl(0, h)] = spec[sl(0, h)]          # DC .. just below Nyquist
        out[sl(m - (h - 1), m)] = spec[sl(h + 1, n)]  # negative frequencies
        # split the real Nyquist mode symmetrically: +k_Ny and -k_Ny
        nyq = spec[sl(h, h + 1)] * 0.5
        out[sl(h, h + 1)] = nyq
        out[sl(m - h, m - h + 1)] = nyq
    return out


def fourier_upsample(field, factor):
    """Spectrally up-sample a real, periodic field by an integer factor per axis.

    Parameters
    ----------
    field : real ndarray
        Samples of a periodic band-limited field on a regular grid.  May have a
        trailing "component" axis (e.g. shape ``(n, n, n, 3)``); only the
        leading ``field.ndim - 1`` axes are up-sampled if ``factor`` is given
        per spatial axis, otherwise pass an explicit list matching ``field.ndim``.
    factor : int or sequence of int
        Integer up-sampling factor, the same for every axis (scalar) or one per
        axis (sequence).  A factor of 1 leaves that axis untouched.

    Returns
    -------
    ndarray
        The up-sampled field; axis ``a`` grows from ``n_a`` to ``factor_a*n_a``.
        Original grid values are reproduced exactly (up to FFT round-off) at the
        coincident coarse nodes.

    Notes
    -----
    For a field with a trailing component axis, pass a per-spatial-axis factor
    list of length ``ndim`` with a trailing ``1`` (e.g. ``[2, 2, 2, 1]``), or a
    scalar -- a scalar only touches axes whose length should grow, so for safety
    the component axis should be handled by :func:`upsample_mesh`.
    """
    field = np.asarray(field)
    factors = _normalize_factors(
        factor if not np.isscalar(factor) else [factor] * field.ndim, field.ndim
    )

    spec = np.fft.fftn(field, axes=tuple(range(field.ndim)))
    scale = 1.0
    for axis, f in enumerate(factors):
        if f != 1:
            spec = _pad_axis(spec, axis, f)
            scale *= f
    out = np.fft.ifftn(spec, axes=tuple(range(field.ndim))) * scale
    return out.real


def upsample_mesh(mesh, factor):
    """Return a new :class:`~lpss.lagrangian.LagrangianMesh` up-sampled by ``factor``.

    The displacement (and velocity, if present) fields are spectrally refined to
    a ``(factor*n)**3`` grid; positions and velocities of the finer mesh are then
    available through the usual :class:`LagrangianMesh` accessors.

    Parameters
    ----------
    mesh : LagrangianMesh
    factor : int or sequence of 3 ints
        Per-dimension integer up-sampling factor.

    Examples
    --------
    >>> fine = upsample_mesh(mesh, 2)          # 2x per dimension -> 8x particles
    >>> xfine = fine.positions_flat()          # up-sampled positions
    >>> vfine = fine.velocities_flat()         # up-sampled velocities
    """
    from .lagrangian import LagrangianMesh

    factors = _normalize_factors(factor, 3)
    spatial = factors + [1]  # never touch the trailing 3-vector component axis

    disp = fourier_upsample(mesh.disp, spatial)
    vel = None
    if mesh.vel is not None:
        vel = fourier_upsample(mesh.vel, spatial)
    return LagrangianMesh(disp=disp.astype(mesh.disp.dtype), boxsize=mesh.boxsize, vel=vel)
