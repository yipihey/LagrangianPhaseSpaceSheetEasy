"""Minimal, dependency-free reader for Gadget-2 ``SnapFormat=1`` snapshots.

The original notebook used :mod:`yt` only to pull the particle ``Coordinates``,
``Velocities`` and ``ParticleIDs`` out of a Gadget-2 file.  ``yt`` is a large
dependency that is awkward to install in a teaching setting, so we provide a
tiny self-contained reader instead.  It understands the classic (un-named,
"format 1") Gadget-2 binary layout:

    [ 256-byte header ]  [ POS ]  [ VEL ]  [ IDS ]  ...

each block being wrapped by a leading and trailing 4-byte Fortran record
marker.  Only the blocks we need are decoded; the rest are skipped.

References
----------
Springel (2005), "The cosmological simulation code GADGET-2",
https://wwwmpa.mpa-garching.mpg.de/gadget/
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

__all__ = ["GadgetHeader", "GadgetSnapshot", "read_gadget"]

_HEADER_SIZE = 256


@dataclass
class GadgetHeader:
    """The fields of the 256-byte Gadget-2 header that we care about."""

    npart: np.ndarray          # particles of each of the 6 types in this file
    mass: np.ndarray           # per-type particle mass (0 => stored in a MASS block)
    time: float                # scale factor a for cosmological runs
    redshift: float
    npart_total: np.ndarray    # particles of each type across all files
    num_files: int
    boxsize: float
    omega0: float
    omega_lambda: float
    hubble_param: float

    @property
    def ntot(self) -> int:
        return int(self.npart.sum())


def _read_record(f):
    """Read one Fortran-wrapped record and return its raw ``bytes``."""
    raw = f.read(4)
    if len(raw) < 4:
        raise EOFError("unexpected end of file while reading a block marker")
    size = struct.unpack("<i", raw)[0]
    data = f.read(size)
    trailing = struct.unpack("<i", f.read(4))[0]
    if trailing != size:
        raise ValueError(
            f"corrupt Gadget block: leading marker {size} != trailing {trailing}"
        )
    return data


def _parse_header(data: bytes) -> GadgetHeader:
    npart = np.frombuffer(data[0:24], dtype="<i4").copy()
    mass = np.frombuffer(data[24:72], dtype="<f8").copy()
    time, redshift = struct.unpack("<2d", data[72:88])
    npart_total = np.frombuffer(data[96:120], dtype="<i4").copy()
    num_files = struct.unpack("<i", data[124:128])[0]
    boxsize = struct.unpack("<d", data[128:136])[0]
    omega0, omega_lambda, hubble = struct.unpack("<3d", data[136:160])
    return GadgetHeader(
        npart=npart,
        mass=mass,
        time=time,
        redshift=redshift,
        npart_total=npart_total,
        num_files=num_files,
        boxsize=boxsize,
        omega0=omega0,
        omega_lambda=omega_lambda,
        hubble_param=hubble,
    )


@dataclass
class GadgetSnapshot:
    """Particle data read from a Gadget-2 snapshot, sorted by particle ID.

    Attributes
    ----------
    pos, vel : (N, 3) float arrays
        Comoving positions and velocities.
    ids : (N,) integer array
        Particle IDs (after sorting these are ``0..N-1`` for a standard
        grid-based set of initial conditions).
    header : GadgetHeader
    """

    pos: np.ndarray
    vel: np.ndarray
    ids: np.ndarray
    header: GadgetHeader

    @property
    def boxsize(self) -> float:
        return self.header.boxsize

    @property
    def mass(self) -> np.ndarray:
        """Per-particle mass for the (single populated) particle type."""
        m = self.header.mass
        nonzero = np.nonzero(self.header.npart)[0]
        # gadget uses one dominant type in these teaching snapshots
        mval = m[nonzero[0]] if nonzero.size else 1.0
        return np.full(self.pos.shape[0], mval)


def read_gadget(filename, sort=True, dtype=np.float64):
    """Read a single-file Gadget-2 (format 1) snapshot.

    Parameters
    ----------
    filename : str or path
    sort : bool, default True
        Sort particles by their IDs.  For grid initial conditions this puts
        the particles back into Lagrangian order, which is exactly what the
        phase-space-sheet construction needs.
    dtype : numpy dtype, default float64
        Output dtype for positions/velocities (the file stores float32).

    Returns
    -------
    GadgetSnapshot
    """
    with open(filename, "rb") as f:
        header = _parse_header(_read_record(f))
        n = header.ntot
        pos = np.frombuffer(_read_record(f), dtype="<f4").reshape(n, 3).astype(dtype)
        vel = np.frombuffer(_read_record(f), dtype="<f4").reshape(n, 3).astype(dtype)
        ids_raw = _read_record(f)

    # IDs may be 32- or 64-bit depending on how the run was compiled.
    if len(ids_raw) == n * 4:
        ids = np.frombuffer(ids_raw, dtype="<u4").astype(np.int64)
    elif len(ids_raw) == n * 8:
        ids = np.frombuffer(ids_raw, dtype="<u8").astype(np.int64)
    else:
        raise ValueError("ID block size is incompatible with 32- or 64-bit IDs")

    if sort:
        order = np.argsort(ids)
        pos = pos[order]
        vel = vel[order]
        ids = ids[order]

    return GadgetSnapshot(pos=pos, vel=vel, ids=ids, header=header)
