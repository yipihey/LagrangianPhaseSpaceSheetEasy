"""The :class:`LagrangianMesh` -- the central object of the library.

Everything in the phase-space-sheet picture starts from a single idea: the dark
matter occupies a *3-dimensional sheet* in 6-dimensional phase space.  Before
any structure forms, every particle sits on a regular grid of **Lagrangian
coordinates** ``q``.  As the simulation evolves, each particle moves to an
**Eulerian position**

    x(q) = q + Psi(q)

where ``Psi`` is the *displacement field*.  Because the sheet cannot tear, the
connectivity of the grid is preserved for all time -- and that connectivity is
exactly what lets us tessellate space into tetrahedra and compute densities,
velocities and caustics.

The displacement field ``Psi`` (and, optionally, the velocity field ``v``) is
all we ever need.  It is *periodic and smooth* even when the positions ``x``
have wrapped many times around the box, which is precisely why it -- and not the
raw positions -- is the right thing to interpolate, Fourier-upsample, or refine.

A :class:`LagrangianMesh` can be built three ways:

* :meth:`LagrangianMesh.zeldovich` -- a fully self-contained analytic field
  (no data required); the recommended pedagogical starting point.
* :meth:`LagrangianMesh.from_positions` -- from a set of particle positions
  laid out in Lagrangian (grid) order, e.g. a simulation snapshot.
* :meth:`LagrangianMesh.from_gadget` -- convenience wrapper around the above
  for a Gadget-2 snapshot read by :mod:`lpss.gadget`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LagrangianMesh", "minimum_image"]


def minimum_image(delta, boxsize):
    """Wrap separation vectors into ``[-L/2, L/2)`` (the periodic minimum image).

    This single operation is what frees us from ever worrying about particles
    that have crossed the periodic boundary: the *separation* between
    neighbouring sheet vertices is always small and continuous even when their
    absolute positions are on opposite sides of the box.
    """
    return delta - boxsize * np.round(delta / boxsize)


def _lagrangian_grid(shape, boxsize, dtype=np.float64):
    """Cell-centred regular grid of ``q`` coordinates, shape ``(*shape, 3)``.

    ``shape`` may be a single int (cubic) or a 3-tuple ``(nx, ny, nz)``.  The box
    is cubic of side ``boxsize``; the grid spacing along each axis is
    ``boxsize / n_axis``, so axes may have different numbers of cells.
    """
    if np.isscalar(shape):
        shape = (int(shape),) * 3
    nx, ny, nz = shape
    xa = boxsize * (0.5 + np.arange(nx, dtype=dtype)) / nx
    ya = boxsize * (0.5 + np.arange(ny, dtype=dtype)) / ny
    za = boxsize * (0.5 + np.arange(nz, dtype=dtype)) / nz
    q = np.empty((nx, ny, nz, 3), dtype=dtype)
    q[..., 0] = xa[:, None, None]
    q[..., 1] = ya[None, :, None]
    q[..., 2] = za[None, None, :]
    return q


@dataclass
class LagrangianMesh:
    """A Lagrangian sheet sampled on an ``n x n x n`` grid.

    Attributes
    ----------
    disp : (n, n, n, 3) float array
        The displacement field ``Psi(q)``.
    boxsize : float
    vel : (n, n, n, 3) float array or None
        Optional velocity field sampled on the same grid.
    """

    disp: np.ndarray
    boxsize: float
    vel: np.ndarray | None = None

    def __post_init__(self):
        if self.disp.ndim != 4 or self.disp.shape[-1] != 3:
            raise ValueError("disp must have shape (nx, ny, nz, 3)")

    # ------------------------------------------------------------------ #
    # basic geometry
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple:
        """Grid dimensions ``(nx, ny, nz)``."""
        return self.disp.shape[:3]

    @property
    def npart(self) -> int:
        return int(np.prod(self.shape))

    @property
    def n(self) -> int:
        """Grid size per dimension (only meaningful for a cubic mesh)."""
        return self.disp.shape[0]

    @property
    def is_cubic(self) -> bool:
        nx, ny, nz = self.shape
        return nx == ny == nz

    @property
    def cell(self) -> float:
        """Lagrangian grid spacing along x (mean inter-particle separation)."""
        return self.boxsize / self.shape[0]

    def lagrangian_coordinates(self):
        """Return the regular grid of ``q`` coordinates, shape ``(*shape, 3)``."""
        return _lagrangian_grid(self.shape, self.boxsize, dtype=self.disp.dtype)

    def positions(self, wrap=True):
        """Eulerian positions ``x = q + Psi``, shape ``(n, n, n, 3)``.

        With ``wrap=True`` the positions are folded back into ``[0, L)``.
        """
        x = self.lagrangian_coordinates() + self.disp
        if wrap:
            x = np.mod(x, self.boxsize)
        return x

    def positions_flat(self, wrap=True):
        """Eulerian positions as a flat ``(n**3, 3)`` array."""
        return self.positions(wrap=wrap).reshape(-1, 3)

    def velocities_flat(self):
        if self.vel is None:
            return None
        return self.vel.reshape(-1, 3)

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_positions(cls, pos, boxsize, n=None, vel=None, fix_orientation=True):
        """Build a mesh from particle positions in Lagrangian (grid) order.

        Parameters
        ----------
        pos : (N, 3) array
            Particle positions, ordered so that ``pos[(i*n + j)*n + k]`` is the
            particle that started at Lagrangian site ``(i, j, k)``.  For Gadget
            grid initial conditions this is the ID-sorted order.
        boxsize : float
        n : int, optional
            Grid size per dimension; inferred as ``round(N**(1/3))`` if omitted.
        vel : (N, 3) array, optional
            Velocities in the same order.
        fix_orientation : bool, default True
            Gadget stores positions in an order whose handedness may make all
            tetrahedra come out with negative volume.  When enabled we detect
            this from the sign of the total volume and flip the axis ordering of
            the vector components so that under-dense regions have positive
            volume (the physically sensible convention).
        """
        pos = np.asarray(pos, dtype=np.float64)
        npart = pos.shape[0]
        if n is None:
            n = int(round(npart ** (1.0 / 3.0)))
        if n ** 3 != npart:
            raise ValueError(f"{npart} particles is not a perfect cube (n={n})")

        q = _lagrangian_grid(n, boxsize)
        pos3d = pos.reshape(n, n, n, 3)

        vel3d = None
        if vel is not None:
            vel3d = np.asarray(vel, dtype=np.float64).reshape(n, n, n, 3)

        if fix_orientation:
            disp = minimum_image(pos3d - q, boxsize)
            if _total_volume_sign(disp, boxsize) < 0:
                # The grid-index ordering (i,j,k) maps to physical (z,y,x) rather
                # than (x,y,z), so every tetrahedron comes out left-handed.
                # Reverse the component ordering to restore positive volumes in
                # the (single-stream) voids.  No particle actually moves.
                pos3d = pos3d[..., ::-1]
                if vel3d is not None:
                    vel3d = vel3d[..., ::-1].copy()

        disp = minimum_image(pos3d - q, boxsize)
        return cls(disp=np.ascontiguousarray(disp), boxsize=boxsize, vel=vel3d)

    @classmethod
    def from_gadget(cls, snapshot, fix_orientation=True):
        """Build a mesh from a :class:`lpss.gadget.GadgetSnapshot`."""
        return cls.from_positions(
            snapshot.pos,
            boxsize=snapshot.boxsize,
            vel=snapshot.vel,
            fix_orientation=fix_orientation,
        )

    @classmethod
    def zeldovich(
        cls,
        n=64,
        boxsize=100.0,
        growth=1.0,
        n_s=-1.0,
        k_cut=None,
        seed=0,
        velocity_factor=1.0,
        dtype=np.float64,
    ):
        """Generate a self-contained Zel'dovich-approximation displacement field.

        The Zel'dovich approximation (Zel'dovich 1970) is the simplest model of
        gravitational structure formation: particles move on straight lines set
        by the initial gravitational potential,

            x(q, t) = q + D(t) * Psi(q),     Psi(q) = -grad phi(q),

        where the displacement potential ``phi`` is sourced by a Gaussian random
        density field with power spectrum ``P(k)``.  Crank up ``growth`` (the
        linear growth factor ``D``) and the sheet folds over on itself: the map
        ``q -> x`` stops being one-to-one, tetrahedra invert, and **caustics /
        shell-crossing** appear.  That is the entire phenomenology the
        phase-space-sheet method is built to capture -- with no simulation
        required.

        Parameters
        ----------
        n : int
            Particles per dimension.
        boxsize : float
        growth : float
            Linear growth factor ``D``; the amplitude of the displacement.
            Larger values produce more shell-crossing.
        n_s : float
            Slope of the input power spectrum ``P(k) ~ k**n_s``.
        k_cut : float, optional
            Gaussian small-scale cutoff (in the same wavenumber units as the
            grid); defaults to the Nyquist wavenumber so the field is smooth.
        seed : int
            Seed for reproducibility.
        velocity_factor : float
            In the Zel'dovich approximation the velocity is parallel to the
            displacement, ``v = velocity_factor * Psi``.  The default keeps the
            two equal up to this constant.
        """
        rng = np.random.default_rng(seed)

        # Fourier wavenumbers on the q-grid (units: 2*pi / boxsize).
        k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=boxsize / n)
        kx = k1d[:, None, None]
        ky = k1d[None, :, None]
        kz = k1d[None, None, :]
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2[0, 0, 0] = 1.0  # avoid division by zero for the DC mode

        kmag = np.sqrt(k2)
        if k_cut is None:
            k_cut = np.pi * n / boxsize  # Nyquist wavenumber
        power = kmag ** n_s * np.exp(-(kmag / k_cut) ** 2)
        power[0, 0, 0] = 0.0  # no mean displacement

        # White-noise field -> coloured by sqrt(P(k)).
        white = np.fft.fftn(rng.standard_normal((n, n, n)))
        delta_k = white * np.sqrt(power)

        # Zel'dovich displacement: Psi_k = i * k / k^2 * delta_k.
        disp = np.empty((n, n, n, 3), dtype=dtype)
        for axis, kk in enumerate((kx, ky, kz)):
            psi_k = 1j * kk / k2 * delta_k
            disp[..., axis] = np.fft.ifftn(psi_k).real

        # Normalise so the displacement has unit rms per component, then scale
        # by the requested growth factor -- this makes ``growth`` an intuitive
        # "how far have particles moved, in grid cells" knob.
        rms = np.sqrt(np.mean(disp ** 2))
        if rms > 0:
            disp *= (boxsize / n) / rms
        disp *= growth

        vel = velocity_factor * disp.copy()
        return cls(disp=disp.astype(dtype), boxsize=float(boxsize), vel=vel)


def _total_volume_sign(disp, boxsize):
    """Sign of the total signed tetrahedral volume for orientation detection.

    Implemented locally with a cheap single-tetrahedron estimate to avoid a
    circular import with :mod:`lpss.tetrahedra`.
    """
    n = disp.shape[0]
    q = _lagrangian_grid(n, boxsize)
    x = q + disp
    # one tetrahedron per cell using neighbours in +x, +y, +z (periodic)
    o = x
    a = minimum_image(np.roll(x, -1, axis=0) - o, boxsize)
    b = minimum_image(np.roll(x, -1, axis=1) - o, boxsize)
    c = minimum_image(np.roll(x, -1, axis=2) - o, boxsize)
    vol = np.einsum("...i,...i->...", a, np.cross(b, c))
    return np.sign(vol.sum())
