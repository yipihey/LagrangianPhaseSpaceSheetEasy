"""lpss -- Lagrangian Phase-Space Sheet, made easy *and* fast.

A small, well-tested library for the phase-space-sheet method of analysing
dark-matter dynamics (Abel, Hahn & Kaehler 2012; Hahn, Abel & Kaehler 2013).

The pieces fit together like this::

    LagrangianMesh            the displacement field Psi(q) -- the central object
      .zeldovich(...)         build one analytically (no data needed)
      .from_gadget(snap)      build one from a simulation snapshot

    tetrahedra.*              tessellate the sheet into 6 tets/cell and compute
                              volumes, centroids, densities, refined particles
                              (numba-accelerated, parallel, periodic)

    fourier.upsample_mesh     spectrally refine Psi (and v) to a finer grid,
                              giving integer-factor up-sampled x and v

    gadget.read_gadget        dependency-free Gadget-2 snapshot reader

See the accompanying notebook ``LagrangianPhaseSpaceSheetEasy.ipynb`` for a
guided, build-it-up-from-scratch tour.
"""

from __future__ import annotations

from ._backend import HAVE_NUMBA, numba_info
from .gadget import GadgetHeader, GadgetSnapshot, read_gadget
from .lagrangian import LagrangianMesh, minimum_image
from .fourier import fourier_upsample, upsample_mesh
from . import tetrahedra

__version__ = "0.1.0"

__all__ = [
    "HAVE_NUMBA",
    "numba_info",
    "GadgetHeader",
    "GadgetSnapshot",
    "read_gadget",
    "LagrangianMesh",
    "minimum_image",
    "fourier_upsample",
    "upsample_mesh",
    "tetrahedra",
    "__version__",
]
