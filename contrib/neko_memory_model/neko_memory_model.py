#!/usr/bin/env python3
"""Per-rank memory model for Neko, with zero-copy on unified-memory APUs.

The model predicts the steady-state (post-initialisation) memory footprint of
one Neko MPI rank by enumerating the allocations the code actually makes, as a
function of the local element count ``nelv``, the polynomial space ``lx`` and
the case configuration.  Every term carries the source location it was read
from, so the model can be audited against the code it describes.

Why a separate accounting for host, mapped and device memory
------------------------------------------------------------
Neko allocates in three ways:

``allocate`` only
    A host array that is never handed to the device.  Mesh connectivity, the
    dofmap's global ids, MPI bookkeeping.

``device_map`` (host array + device pointer)
    The overwhelming majority of the footprint.  Normally this costs *twice*
    the array: the Fortran allocation plus a ``hipMalloc`` replica.  With
    ``NEKO_HIP_ZEROCOPY=1`` on a unified-memory APU the device pointer aliases
    the host allocation instead, so it costs the array *once*
    (``src/device/hip/unified.hip:165`` ``hip_map``).

``device_alloc`` only
    Device-resident buffers with no host array -- the gather-scatter exchange
    buffers.  Zero-copy does not change these.

On a discrete GPU the host and device halves come out of different pools.  On
an APU such as MI300A there is one physical pool, so the number that matters
is ``host + mapped*(1 or 2) + device``, and halving the mapped half is a
direct increase in the problem size that fits.

Usage
-----
    ./neko_memory_model.py --elements 8000 --lx 8 --budget 120GiB
    ./neko_memory_model.py --elements 8000 --lx 8 --no-zero-copy
    ./neko_memory_model.py --elements 8000 --lx 8 --real-type sp
    ./neko_memory_model.py --fit --budget 120GiB --lx 8
    ./neko_memory_model.py --elements 8000 --lx 8 --json

Defaults describe the configuration this model was written for: double
precision (``--enable-real=dp``), dealiasing on, DNS (no LES), CG + Jacobi for
velocity, GMRES + PHMG (TreeAMG coarse grid) for pressure, no solution
projection on either, and ``fluid_stats`` with the full 44-field set written
as a full 3D field.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model constants read off the source
# ---------------------------------------------------------------------------

#: Arrays of size ``lx*ly*lz*nelv`` held by a ``COEF_FULL`` coef_t, all
#: device-mapped: G11 G22 G33 G12 G13 G23, dxdr..dzdt, drdx..dtdz, jac,
#: jacinv, B, Binv, h1, h2, mult.  Blag/Blaglag are pointers to B.
#: src/sem/coef.f90:327-378
COEF_FULL_N_ARRAYS = 31

#: Arrays of size ``lx*ly*6*nelv`` held by a ``COEF_FULL`` coef_t:
#: area, nx, ny, nz.  src/sem/coef.f90:361-366
COEF_FULL_FACET_ARRAYS = 4

#: What survives ``coef_release_scratch`` under ``COEF_OPERATOR``:
#: G11..G23, B, h1, h2, mult.  src/sem/coef.f90:807-940
COEF_OPERATOR_N_ARRAYS = 10

#: Arrays allocated by the geometry-only ``coef_init_empty`` used for the
#: dealiasing space: drdx..dtdz.  src/sem/coef.f90:250-292
COEF_EMPTY_N_ARRAYS = 9

#: Work arrays of size ``nelv*lxd^3`` in adv_dealias_t: temp, tbf, tx, ty,
#: tz, vr, vs, vt.  src/fluid/bcknd/advection/adv_dealias.f90:135-153
DEALIAS_WORK_ARRAYS = 8

#: Persistent fields of size ``n`` for an incompressible pnpn fluid:
#:   registry: u v w p u_e v_e w_e mu mu_tot rho            (10)
#:   lag series: ulag(2) vlag(2) wlag(2)                     (6)
#:   rhs: f_x f_y f_z                                        (3)
#:   pnpn: p_res u_res v_res w_res, dp du dv dw,
#:         abx1 aby1 abz1 abx2 aby2 abz2, advx advy advz     (17)
#: src/fluid/fluid_scheme_incompressible.f90:206,278-281,325-331,681-683
#: src/fluid/fluid_pnpn.f90:268,343-361
FLUID_FIELDS = 10 + 6 + 3 + 17

#: fluid_stats_t with ``set_of_stats = full``: 44 registered mean fields, the
#: five work fields (stats_work, stats_u, stats_v, stats_w, stats_p) and the
#: nine velocity-gradient fields.  src/fluid/fluid_stats.f90:188-266
STATS_FULL_FIELDS = 44 + 5 + 9
#: ``set_of_stats = basic``: 11 mean fields + the five work fields.
STATS_BASIC_FIELDS = 11 + 5

#: Peak number of scratch fields held at once, set by the pnpn pressure
#: residual (ta1..ta3, wa1..wa3, work1, work2).
#: src/fluid/bcknd/device/pnpn_res_device.F90:303-310
SCRATCH_PEAK_FIELDS = 8

#: Krylov workspaces, in multiples of ``n``.
KSP_WORK_VECTORS = {
    "cg": 4,          # w r p z    src/krylov/bcknd/device/cg_device.f90:86
    "pipecg": 10,     # p q r s u w z mi ni ar
    "bicgstab": 8,
    "cheby": 3,       # d w r
    "gmres": None,    # 2 + 2*m_restart, see GMRES_RESTART
}

#: GMRES restart depth.  Hard-coded, not settable from the case file.
#: src/krylov/bcknd/device/gmres_device.F90:65
GMRES_RESTART = 30

#: Preconditioner workspaces, in multiples of ``n``.
PC_WORK_VECTORS = {
    "ident": 0,
    "jacobi": 1,      # d    src/krylov/bcknd/device/pc_jacobi_device.F90:121
}

#: Per PHMG level: r, w, z + the Jacobi diagonal + the Chebyshev d, w, r.
#: src/multigrid/phmg.f90:256-258,286,299-301
PHMG_LEVEL_VECTORS = 3 + PC_WORK_VECTORS["jacobi"] + KSP_WORK_VECTORS["cheby"]

#: Default p-coarsening schedule, giving levels lx, 4, 2.
#: src/multigrid/phmg.f90:163-166 (``pcoarsening_schedule`` = [3, 1], +1)
PHMG_DEFAULT_SCHEDULE = (3, 1)

#: Default number of TreeAMG levels under PHMG.  src/multigrid/phmg.f90:151-152
TAMG_DEFAULT_LEVELS = 3

#: Greedy aggregation target: one aggregate per eight elements.
#: src/multigrid/tree_amg_multigrid.f90:148
TAMG_AGGREGATION_RATIO = 8

#: Components in the fused vector gather-scatter buffer.  src/gs/gs_comm.f90:50
GS_VEC_NC = 3

#: Bytes per slot of a Neko hash table.  A slot is an ``h_tuple_t``: two
#: logicals, two unlimited-polymorphic allocatable descriptors (24 B each on
#: gfortran) and two list pointers, so 72 B inline -- plus the two separate
#: heap allocations the descriptors point at, which a 32 B minimum malloc
#: chunk rounds up to 64 B together.  Compiler- and allocator-dependent.
#: src/adt/htable.f90:62-70,275-288
HTABLE_BYTES_PER_SLOT = 136

#: Load factor a table reaches before quadratic probing gives up and doubles
#: it.  src/adt/htable.f90:349,400
HTABLE_LOAD_FACTOR = 0.6

#: Per-element bytes of mesh connectivity retained after ``generate_conn``:
#: the polymorphic hex_t in elements(:) (~160 B), pt_lid 32, edge_lid 48,
#: face_lid 24, facet_neigh 24, facet_type 12, dfrmd_el 4.
#: src/mesh/mesh.f90:88-137,278-308
MESH_BYTES_PER_ELEMENT = 160 + 32 + 48 + 24 + 24 + 12 + 4

#: point_t is three doubles plus an id; the emptied point_neigh stack keeps
#: its descriptor.  src/mesh/mesh.f90:88,132,410-415
MESH_BYTES_PER_POINT = 32 + 72

#: Unique facets in a hex mesh: six per element, interior ones shared by two.
MESH_FACETS_PER_ELEMENT = 3.0

#: Working precisions selectable at configure time with ``--enable-real``,
#: mapped to the byte width of ``rp`` (the working real) and ``xp`` (the
#: extended real used for accumulations).  ``dp`` is the default.
#: configure.ac:22-28,195-253, src/config/num_types.f90.in
REAL_TYPES = {
    "ssp": {"rp": 4, "xp": 4,
            "desc": "rp = REAL32, xp = REAL32"},
    "sp": {"rp": 4, "xp": 8,
           "desc": "rp = REAL32, xp = REAL64"},
    "dp": {"rp": 8, "xp": 8,
           "desc": "rp = REAL64, xp = REAL64 (configure default)"},
    "qp": {"rp": 16, "xp": 16,
           "desc": "rp = REAL128, xp = REAL128"},
}

SI = {"B": 1, "kB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
      "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """A Neko case, as far as memory is concerned."""

    nelv: int = 8000
    lx: int = 8
    lxd: int | None = None           # None -> (3*lx)//2, Neko's default

    #: Working precision, as passed to ``configure --enable-real``.
    real_type: str = "dp"

    zero_copy: bool = True

    dealias: bool = True
    les: bool = False                # DNS by default

    vel_solver: str = "cg"
    vel_precon: str = "jacobi"
    prs_solver: str = "gmres"
    prs_precon: str = "phmg"

    phmg_schedule: tuple[int, ...] = PHMG_DEFAULT_SCHEDULE
    tamg_levels: int = TAMG_DEFAULT_LEVELS

    prs_projection_dim: int = 0      # "no projection"
    vel_projection_dim: int = 0

    stats: str = "full"              # none | basic | full
    flow_rate_force: bool = False

    scratch_fields: int = SCRATCH_PEAK_FIELDS

    #: Fraction of this rank's gather-scatter entries that are shared with
    #: another rank.  ``None`` derives it from a cube-shaped subdomain.
    shared_fraction: float | None = None

    #: Fraction of the rank's element facets carrying a strong boundary
    #: condition, and the number of distinct bc objects holding a mask.
    bc_facet_fraction: float = 0.02
    bc_objects: int = 4

    def __post_init__(self) -> None:
        if self.real_type not in REAL_TYPES:
            raise ValueError(
                f"unknown --enable-real value '{self.real_type}'; "
                f"expected one of {', '.join(REAL_TYPES)}")
        if self.lxd is None:
            self.lxd = (3 * self.lx) // 2

    # -- derived sizes ------------------------------------------------------

    @property
    def rp_bytes(self) -> int:
        """Bytes per working real, set by ``configure --enable-real``."""
        return REAL_TYPES[self.real_type]["rp"]

    @property
    def xp_bytes(self) -> int:
        """Bytes per extended real.

        Only ever backs fixed-size arrays -- the CPU GMRES Hessenberg and
        rotation coefficients, and the device reduction buffers -- so it does
        not enter the footprint at problem scale.  ``ssp`` and ``sp`` are
        therefore the same size.
        """
        return REAL_TYPES[self.real_type]["xp"]

    @property
    def n(self) -> int:
        """Local degrees of freedom, ``nelv * lx^3``."""
        return self.nelv * self.lx ** 3

    @property
    def n_facet(self) -> int:
        """Facet points, ``nelv * lx^2 * 6``."""
        return self.nelv * self.lx ** 2 * 6

    @property
    def n_dealias(self) -> int:
        """Dealiasing-space dofs, ``nelv * lxd^3``."""
        return self.nelv * self.lxd ** 3

    def gs_entries(self, lx: int | None = None) -> int:
        """Gather-scatter entries: every non-interior point of every element.

        ``lx^3 - (lx-2)^3`` = ``8 + 12(lx-2) + 6(lx-2)^2`` vertex, edge and
        facet points per element, each pushed once by ``gs_init_mapping``.
        src/gs/gather_scatter.f90:654-700
        """
        lx = self.lx if lx is None else lx
        return self.nelv * (lx ** 3 - max(lx - 2, 0) ** 3)

    def gs_shared(self, lx: int | None = None) -> int:
        """Entries shared with another rank.

        Derived, unless overridden, from a cube-shaped subdomain of ``nelv``
        elements: its six faces carry ``6 * nelv^(2/3) * lx^2`` dofs.
        """
        lx = self.lx if lx is None else lx
        if self.shared_fraction is not None:
            return int(self.shared_fraction * self.gs_entries(lx))
        faces = 6.0 * self.nelv ** (2.0 / 3.0) * lx ** 2
        return int(min(faces, self.gs_entries(lx)))

    def gs_blocks(self, lx: int | None = None) -> int:
        """Non-facet gather-scatter blocks, ``nblks`` in ``gs_find_blks``.

        Vertices are shared by up to eight elements and edge points by up to
        four, so the vertex+edge entries collapse to roughly
        ``nelv * (1 + 3*(lx-2))`` distinct ids.
        src/gs/gather_scatter.f90:1429-1467
        """
        lx = self.lx if lx is None else lx
        return self.nelv * (1 + 3 * max(lx - 2, 0))

    @property
    def phmg_levels(self) -> list[int]:
        """``lx`` of each PHMG level, finest first."""
        return [self.lx] + [s + 1 for s in self.phmg_schedule]

    @property
    def bc_dofs(self) -> int:
        return int(self.bc_facet_fraction * self.n_facet)


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

#: Setup phases in the order ``fluid_pnpn_init`` walks them, used to place
#: transient allocations against what is already resident when they happen.
#: src/fluid/fluid_scheme_incompressible.f90:190-340,
#: src/fluid/fluid_pnpn.f90:243-430, src/case.f90 (simcomps last)
PHASES = [
    "mesh",
    "function space and dofmap",
    "gather-scatter",
    "coefficients",
    "fluid fields",
    "velocity solver",
    "pressure solver",
    "advection",
    "boundary conditions and scratch",
    "statistics and i/o",
]


@dataclass
class Term:
    """One line of the model."""

    group: str
    name: str
    mapped: int = 0        # device_map: host array + device alias/replica
    host: int = 0          # allocate only
    device: int = 0        # device_alloc only
    formula: str = ""
    source: str = ""
    exact: bool = True     # False => the size rests on a modelling assumption
    phase: int = 0         # index into PHASES: when it is allocated
    transient: bool = False  # freed again before the time loop starts

    def total(self, zero_copy: bool) -> int:
        return self.host + self.device + self.mapped * (1 if zero_copy else 2)


def next_pow2(x: float) -> int:
    return 1 << max(int(math.ceil(math.log2(max(x, 4.0)))), 2)


def htable_bytes(entries: float, initial_slots: float) -> int:
    """Bytes held by a Neko hash table.

    ``htable_init`` rounds the requested size up to a power of two, and
    ``htable_set`` doubles the table whenever quadratic probing fails to place
    a key, which happens around a load factor of ``HTABLE_LOAD_FACTOR``.
    src/adt/htable.f90:273-288,349-400
    """
    slots = max(next_pow2(initial_slots),
                next_pow2(entries / HTABLE_LOAD_FACTOR))
    return slots * HTABLE_BYTES_PER_SLOT


def _dofmap(cfg: Config, n: int, label: str, phase: int) -> list[Term]:
    """x, y, z mapped; the global ids and the shared flags host-only."""
    return [
        Term("dofmap", f"{label}: coordinates x, y, z",
             mapped=3 * n * cfg.rp_bytes, phase=phase,
             formula="3 * n * rp", source="src/sem/dofmap.f90:137-151"),
        Term("dofmap", f"{label}: global ids + shared flags",
             host=n * (8 + 4), phase=phase,
             formula="n * (8 + 4)", source="src/sem/dofmap.f90:117-118"),
    ]


def _gather_scatter(cfg: Config, lx: int, label: str,
                    phase: int) -> list[Term]:
    m = cfg.gs_entries(lx)
    m_s = cfg.gs_shared(lx)
    m_l = m - m_s
    nb = cfg.gs_blocks(lx)
    n_lvl = cfg.nelv * lx ** 3
    rp = cfg.rp_bytes

    terms = [
        # local_gs (rp) + local_dof_gs (i4) + local_gs_dof (i4)
        Term("gather-scatter", f"{label}: local gather/scatter maps",
             mapped=m_l * (rp + 4 + 4), phase=phase,
             formula="m_local * (rp + 8)",
             source="src/gs/bcknd/device/gs_device.F90:277-306"),
        # shared_gs (rp) + shared_gs_v (3 rp) + shared_dof_gs + shared_gs_dof
        Term("gather-scatter", f"{label}: shared gather/scatter maps",
             mapped=m_s * (rp * (1 + GS_VEC_NC) + 4 + 4), phase=phase,
             formula="m_shared * ((1 + GS_VEC_NC) * rp + 8)",
             source="src/gs/gather_scatter.f90:1341-1351, "
                    "src/gs/bcknd/device/gs_device.F90:335-364"),
        Term("gather-scatter", f"{label}: block length/offset tables",
             mapped=2 * nb * 4, phase=phase,
             formula="2 * nblks * 4", exact=False,
             source="src/gs/gather_scatter.f90:1429-1467"),
        # gs_device_mpi: buf_d (rp) + buf_v_d (3 rp) + dof_d (i4), send & recv
        Term("gather-scatter", f"{label}: MPI exchange buffers (device)",
             device=2 * m_s * (rp * (1 + GS_VEC_NC) + 4), phase=phase,
             formula="2 * m_shared * ((1 + GS_VEC_NC) * rp + 4)", exact=False,
             source="src/gs/bcknd/device/gs_device_mpi.F90:264-277"),
        Term("gather-scatter", f"{label}: MPI peer dof lists (host)",
             host=2 * m_s * 4, phase=phase,
             formula="2 * m_shared * 4", exact=False,
             source="src/gs/bcknd/device/gs_device_mpi.F90:279"),
        # Freed at the end of gs_schedule / on return from gs_init_mapping.
        Term("gather-scatter", f"{label}: shared-dof hash table (setup only)",
             host=htable_bytes(m_s, n_lvl), phase=phase, transient=True,
             formula="htable(m_shared entries, n initial slots)", exact=False,
             source="src/gs/gather_scatter.f90:633,1618"),
        Term("gather-scatter", f"{label}: local-dof hash table (setup only)",
             host=htable_bytes(m_l / 4.0, n_lvl / max(lx, 1)),
             phase=phase, transient=True,
             formula="htable(~m_local/4 entries, n/lx initial slots)",
             exact=False, source="src/gs/gather_scatter.f90:629"),
        Term("gather-scatter", f"{label}: mapping stacks (setup only)",
             host=4 * m * 2 * 2, phase=phase, transient=True,
             formula="2 stacks per entry * 4 B, x2 for stack doubling",
             exact=False, source="src/gs/gather_scatter.f90:636-646"),
    ]
    return terms


def _coef_full(cfg: Config) -> list[Term]:
    rp = cfg.rp_bytes
    return [
        Term("coefficients", "c_Xh: volume coefficients (COEF_FULL)",
             mapped=COEF_FULL_N_ARRAYS * cfg.n * rp, phase=3,
             formula=f"{COEF_FULL_N_ARRAYS} * n * rp",
             source="src/sem/coef.f90:327-378"),
        Term("coefficients", "c_Xh: facet area and normals",
             mapped=COEF_FULL_FACET_ARRAYS * cfg.n_facet * rp, phase=3,
             formula=f"{COEF_FULL_FACET_ARRAYS} * n_facet * rp",
             source="src/sem/coef.f90:361-366"),
    ]


def _tamg(cfg: Config, lx_coarse: int) -> list[Term]:
    """TreeAMG, used by PHMG as its coarse-grid solver.

    Level 1 aggregates the ``nelv * lx_coarse^3`` dofs of the coarsest PHMG
    space into one aggregate per element; each further level aggregates
    greedily by a factor of eight.
    """
    rp = cfg.rp_bytes
    n_c = cfg.nelv * lx_coarse ** 3

    # fine_lvl_dofs of each AMG level: the dof count the level restricts from.
    dofs: list[int] = [n_c]
    nodes: list[int] = [cfg.nelv]
    for _ in range(1, cfg.tamg_levels):
        dofs.append(nodes[-1])
        nodes.append(max(nodes[-1] // TAMG_AGGREGATION_RATIO, 1))
    total_dofs = sum(dofs)

    terms = [
        # tamg_lvl_t: wrk_in, wrk_out
        Term("pressure preconditioner", "TreeAMG: level work vectors",
             mapped=2 * total_dofs * rp,
             formula="2 * sum(dofs_l) * rp",
             phase=6, source="src/multigrid/tree_amg.f90:215-225"),
        # amg_cheby_t d, w, r + tamg_wrk_t r, b, x
        Term("pressure preconditioner", "TreeAMG: smoother and cycle vectors",
             mapped=6 * total_dofs * rp,
             formula="6 * sum(dofs_l) * rp",
             phase=6, source="src/multigrid/tree_amg_smoother.f90:119-125, "
                    "src/multigrid/tree_amg_multigrid.f90:191-208"),
        # map_f2c(0:ndofs) per level
        Term("pressure preconditioner", "TreeAMG: fine-to-coarse maps",
             mapped=sum(d + 1 for d in dofs) * 4,
             formula="sum(dofs_l + 1) * 4",
             phase=6, source="src/multigrid/tree_amg.f90:209-213"),
        # map_finest2lvl(0:n_finest) for every level
        Term("pressure preconditioner", "TreeAMG: finest-level maps",
             mapped=cfg.tamg_levels * (n_c + 1) * 4,
             formula="n_amg_levels * (n_coarse + 1) * 4",
             phase=6, source="src/multigrid/tree_amg.f90:165-168"),
        # tamg_node_t: dofs(i4) + interp_r(rp) + interp_p(rp) per dof
        Term("pressure preconditioner", "TreeAMG: aggregation tree (host)",
             host=total_dofs * (4 + 2 * rp),
             formula="sum(dofs_l) * (4 + 2 * rp)",
             phase=6, source="src/multigrid/tree_amg.f90:288-292"),
        # agg_ptr(nagg+1) + agg_dof(n_finest) per level
        Term("pressure preconditioner", "TreeAMG: aggregate CSR (host)",
             host=cfg.tamg_levels * n_c * 4 + sum(nodes) * 4,
             formula="n_amg_levels * n_coarse * 4 + sum(naggs_l) * 4",
             phase=6, source="src/multigrid/tree_amg_multigrid.f90:654-655"),
    ]
    return terms


def _phmg(cfg: Config) -> list[Term]:
    rp = cfg.rp_bytes
    levels = cfg.phmg_levels
    terms: list[Term] = []

    # Level 0 shares the fluid's space, dofmap, gs and coef.
    terms.append(
        Term("pressure preconditioner",
             f"PHMG level 0 (lx={levels[0]}): r, w, z, Jacobi, Chebyshev",
             mapped=PHMG_LEVEL_VECTORS * cfg.n * rp,
             formula=f"{PHMG_LEVEL_VECTORS} * n * rp",
             phase=6, source="src/multigrid/phmg.f90:256-258,286,299-301"))

    for lx_i in levels[1:]:
        n_i = cfg.nelv * lx_i ** 3
        tag = f"PHMG level lx={lx_i}"
        terms.append(
            Term("pressure preconditioner",
                 f"{tag}: r, w, z, Jacobi, Chebyshev",
                 mapped=PHMG_LEVEL_VECTORS * n_i * rp,
                 formula=f"{PHMG_LEVEL_VECTORS} * n_lvl * rp",
                 phase=6, source="src/multigrid/phmg.f90:256-258,286,299-301"))
        terms.append(
            Term("pressure preconditioner",
                 f"{tag}: coefficients (COEF_OPERATOR)",
                 mapped=COEF_OPERATOR_N_ARRAYS * n_i * rp,
                 formula=f"{COEF_OPERATOR_N_ARRAYS} * n_lvl * rp",
                 phase=6, source="src/multigrid/phmg.f90:250-251, "
                 "src/sem/coef.f90:807-940"))
        # coef_init_all allocates the full set and releases the scratch only
        # at the end, so the coarse level briefly costs COEF_FULL.
        terms.append(
            Term("pressure preconditioner",
                 f"{tag}: coefficient scratch released after init",
                 mapped=(COEF_FULL_N_ARRAYS - COEF_OPERATOR_N_ARRAYS)
                 * n_i * rp,
                 phase=6, transient=True,
                 formula=f"({COEF_FULL_N_ARRAYS} - {COEF_OPERATOR_N_ARRAYS})"
                         " * n_lvl * rp",
                 source="src/sem/coef.f90:327-378,807-940"))
        terms.extend(_dofmap(cfg, n_i, tag, 6))
        terms.extend(_gather_scatter(cfg, lx_i, tag, 6))

    terms.extend(_tamg(cfg, levels[-1]))
    return terms


def _solver_vectors(name: str) -> int:
    if name == "gmres":
        return 2 + 2 * GMRES_RESTART
    if name not in KSP_WORK_VECTORS or KSP_WORK_VECTORS[name] is None:
        raise ValueError(f"unknown Krylov solver '{name}'")
    return KSP_WORK_VECTORS[name]


def build_terms(cfg: Config) -> list[Term]:
    """Enumerate every modelled allocation for ``cfg``."""
    rp = cfg.rp_bytes
    n = cfg.n
    terms: list[Term] = []

    # -- mesh ---------------------------------------------------------------
    # Unique points: for a large hex mesh each element contributes about one
    # new vertex, so mpts ~ nelv (8*nelv is the trivial upper bound).
    mpts = cfg.nelv
    terms.append(
        Term("mesh", "elements, point and facet tables",
             host=cfg.nelv * MESH_BYTES_PER_ELEMENT
             + mpts * MESH_BYTES_PER_POINT,
             formula=f"nelv * {MESH_BYTES_PER_ELEMENT} B + mpts * "
                     f"{MESH_BYTES_PER_POINT} B", exact=False,
             source="src/mesh/mesh.f90:88-137,278-308"))
    terms.append(
        Term("mesh", "facet_map hash table",
             host=htable_bytes(MESH_FACETS_PER_ELEMENT * cfg.nelv, cfg.nelv),
             formula="htable(3 * nelv entries, nelv slots)", exact=False,
             source="src/mesh/mesh.f90:284-287"))
    terms.append(
        Term("mesh", "htel element hash table",
             host=htable_bytes(cfg.nelv, cfg.nelv),
             formula="htable(nelv entries, nelv initial slots)", exact=False,
             source="src/mesh/mesh.f90:329"))
    terms.append(
        Term("mesh", "htp point hash table (setup only)",
             host=htable_bytes(mpts, 8 * cfg.nelv),
             transient=True,
             formula="htable(mpts entries, 8 * nelv slots)", exact=False,
             source="src/mesh/mesh.f90:328,357-358"))
    terms.append(
        Term("mesh", "edge_pts and face_pts (setup only)",
             host=cfg.nelv * (2 * 12 * 4 + 4 * 6 * 4),
             transient=True,
             formula="nelv * (2*12 + 4*6) * 4", 
             source="src/mesh/mesh.f90:750,796,714-716"))

    # -- SEM state ----------------------------------------------------------
    terms.extend(_dofmap(cfg, n, "dm_Xh", 1))
    terms.extend(_coef_full(cfg))
    terms.extend(_gather_scatter(cfg, cfg.lx, "gs_Xh", 2))

    # -- fields -------------------------------------------------------------
    terms.append(
        Term("fields", f"fluid fields ({FLUID_FIELDS} x n)",
             mapped=FLUID_FIELDS * n * rp,
             formula=f"{FLUID_FIELDS} * n * rp",
             phase=4,
             source="src/fluid/fluid_scheme_incompressible.f90:206-331, "
                    "src/fluid/fluid_pnpn.f90:343-361"))
    terms.append(
        Term("fields", f"scratch registry ({cfg.scratch_fields} x n)",
             mapped=cfg.scratch_fields * n * rp,
             formula="scratch_peak * n * rp", exact=False, phase=8,
             source="src/fluid/bcknd/device/pnpn_res_device.F90:303-310"))

    # -- advection ----------------------------------------------------------
    if cfg.dealias:
        nd = cfg.n_dealias
        terms.append(
            Term("advection", f"dealias work arrays (lxd={cfg.lxd})",
                 mapped=DEALIAS_WORK_ARRAYS * nd * rp,
                 formula=f"{DEALIAS_WORK_ARRAYS} * n_dealias * rp",
                 phase=7,
                 source="src/fluid/bcknd/advection/adv_dealias.f90:135-153"))
        terms.append(
            Term("advection", f"dealias geometry coef_GL (lxd={cfg.lxd})",
                 mapped=COEF_EMPTY_N_ARRAYS * nd * rp,
                 formula=f"{COEF_EMPTY_N_ARRAYS} * n_dealias * rp", phase=7,
                 source="src/sem/coef.f90:250-292"))
    else:
        terms.append(
            Term("advection", "non-dealiased work fields (6 x n)",
                 mapped=6 * n * rp,
                 formula="6 * n * rp", phase=7,
                 source="src/fluid/bcknd/advection/adv_no_dealias.f90"))

    # -- velocity solve -----------------------------------------------------
    terms.append(
        Term("velocity solver", f"{cfg.vel_solver} work vectors",
             mapped=_solver_vectors(cfg.vel_solver) * n * rp,
             formula=f"{_solver_vectors(cfg.vel_solver)} * n * rp", phase=5,
             source="src/krylov/bcknd/device/cg_device.f90:86-94"))
    if PC_WORK_VECTORS.get(cfg.vel_precon):
        terms.append(
            Term("velocity solver", f"{cfg.vel_precon} preconditioner",
                 mapped=PC_WORK_VECTORS[cfg.vel_precon] * n * rp,
                 formula=f"{PC_WORK_VECTORS[cfg.vel_precon]} * n * rp",
                 phase=5,
                 source="src/krylov/bcknd/device/pc_jacobi_device.F90:121"))

    # -- pressure solve -----------------------------------------------------
    nv = _solver_vectors(cfg.prs_solver)
    terms.append(
        Term("pressure solver", f"{cfg.prs_solver} work vectors"
             + (f" (restart {GMRES_RESTART})"
                if cfg.prs_solver == "gmres" else ""),
             mapped=nv * n * rp,
             formula=(f"(2 + 2*{GMRES_RESTART}) * n * rp"
                      if cfg.prs_solver == "gmres" else f"{nv} * n * rp"),
             phase=6,
             source="src/krylov/bcknd/device/gmres_device.F90:162-189"))

    if cfg.prs_precon == "phmg":
        terms.extend(_phmg(cfg))
    elif PC_WORK_VECTORS.get(cfg.prs_precon):
        terms.append(
            Term("pressure preconditioner", f"{cfg.prs_precon}",
                 mapped=PC_WORK_VECTORS[cfg.prs_precon] * n * rp,
                 formula=f"{PC_WORK_VECTORS[cfg.prs_precon]} * n * rp",
                 phase=6,
                 source="src/krylov/bcknd/device/pc_jacobi_device.F90:121"))

    # -- projection ---------------------------------------------------------
    # xx(n,L), bb(n,L), xbar(n); nothing is allocated when L <= 0.
    # src/common/projection.f90:148-157
    if cfg.prs_projection_dim > 0:
        terms.append(
            Term("projection",
                 f"pressure projection (L={cfg.prs_projection_dim})",
                 mapped=(2 * cfg.prs_projection_dim + 1) * n * rp,
                 formula="(2L + 1) * n * rp", phase=6,
                 source="src/common/projection.f90:154-156"))
    if cfg.vel_projection_dim > 0:
        terms.append(
            Term("projection",
                 f"velocity projection (3 x L={cfg.vel_projection_dim})",
                 mapped=3 * (2 * cfg.vel_projection_dim + 1) * n * rp,
                 formula="3 * (2L + 1) * n * rp", phase=5,
                 source="src/common/projection_vel.f90:74-76"))

    # -- forced flow rate ---------------------------------------------------
    if cfg.flow_rate_force:
        terms.append(
            Term("fields", "flow rate control (u_vol, v_vol, w_vol, p_vol)",
                 mapped=4 * n * rp,
                 formula="4 * n * rp", phase=6,
                 source="src/fluid/fluid_volflow.f90:134-137"))

    # -- statistics ---------------------------------------------------------
    if cfg.stats != "none":
        nfields = (STATS_FULL_FIELDS if cfg.stats == "full"
                   else STATS_BASIC_FIELDS)
        terms.append(
            Term("statistics", f"fluid_stats '{cfg.stats}' ({nfields} x n)",
                 mapped=nfields * n * rp,
                 formula=f"{nfields} * n * rp", phase=9,
                 source="src/fluid/fluid_stats.f90:201-266"))

    # -- boundary conditions ------------------------------------------------
    terms.append(
        Term("boundary conditions", "bc masks (msk, facet_node_msk, facet)",
             mapped=cfg.bc_objects * 3 * (cfg.bc_dofs + 1) * 4,
             formula="n_bc * 3 * (n_bc_dofs + 1) * 4", exact=False, phase=8,
             source="src/bc/bc.f90:512-517"))

    # -- device reduction buffers -------------------------------------------
    # hip_buffer_reserve grows these to the largest reduction ever run and
    # keeps them until device teardown.  Each is a pinned host allocation
    # plus a device allocation; neither is a device_map, so zero-copy leaves
    # both alone.  redbuf_xp is the only xp-typed allocation in the whole
    # footprint that scales with the problem size, which is why `ssp` and
    # `sp` differ at all.
    nb = -(-n // 1024) + 1
    terms.append(
        Term("reductions", "glsc/glsum buffer (rp), pinned host + device",
             host=nb * rp, device=nb * rp, phase=4,
             formula="(ceil(n/1024) + 1) * rp, twice",
             source="src/math/bcknd/device/hip/math.hip:765-767, "
                    "src/device/hip/buffer.hip:47-67"))
    terms.append(
        Term("reductions", "glsc/glsum buffer (xp), pinned host + device",
             host=nb * cfg.xp_bytes, device=nb * cfg.xp_bytes, phase=4,
             formula="(ceil(n/1024) + 1) * xp, twice",
             source="src/math/bcknd/device/hip/math.hip:769-771"))
    terms.append(
        Term("reductions", "CFL reduction buffer (device only)",
             device=cfg.nelv * 8, phase=4,
             formula="nelv * 8 (an explicit double, not rp)",
             source="src/math/bcknd/device/hip/opr_cfl.hip:57,72"))

    # -- I/O ----------------------------------------------------------------
    terms.append(
        Term("i/o", "fld write staging buffer (transient)",
             host=3 * n * 4 + cfg.nelv * 4,
             formula="gdim * n * sp + nelv * 4", exact=False, phase=9,
             source="src/io/fld_file.f90:344-346,287"))

    return terms


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def steady(terms: list[Term]) -> list[Term]:
    """The allocations still resident when the time loop starts."""
    return [t for t in terms if not t.transient]


def totals(terms: list[Term], zero_copy: bool) -> dict[str, int]:
    resident = steady(terms)
    mapped = sum(t.mapped for t in resident)
    host = sum(t.host for t in resident)
    device = sum(t.device for t in resident)
    return {
        "mapped": mapped,
        "host_only": host,
        "device_only": device,
        # One physical pool on an APU.
        "unified_total": host + device + mapped * (1 if zero_copy else 2),
        # For comparison, how the same case splits on a discrete GPU.
        "discrete_device": mapped + device,
        "discrete_host": mapped + host,
    }


def setup_peak(terms: list[Term], zero_copy: bool) -> tuple[int, str]:
    """Highest footprint reached during setup, and what drives it.

    A transient allocation sits on top of everything allocated in its own
    phase and every earlier one, so the peak is the largest such sum.  It can
    exceed the steady-state footprint, which is how a case that would have run
    fine still dies during initialisation.
    """
    resident = sorted(steady(terms), key=lambda t: t.phase)
    running: dict[int, int] = {}
    acc = 0
    for phase in range(len(PHASES)):
        acc += sum(t.total(zero_copy) for t in resident if t.phase == phase)
        running[phase] = acc

    peak, driver = acc, "steady state"
    for t in terms:
        if not t.transient:
            continue
        here = running[t.phase] + t.total(zero_copy)
        if here > peak:
            peak, driver = here, f"{t.name} (during '{PHASES[t.phase]}')"
    return peak, driver


def humanize(nbytes: float) -> str:
    for unit, scale in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20),
                        ("KiB", 2**10)):
        if abs(nbytes) >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes:.0f} B"


def parse_size(text: str) -> int:
    text = text.strip()
    for unit in sorted(SI, key=len, reverse=True):
        if text.endswith(unit):
            return int(float(text[: -len(unit)]) * SI[unit])
    return int(float(text))


def report(cfg: Config, terms: list[Term], budget: int | None,
           show_terms: bool) -> str:
    tot = totals(terms, cfg.zero_copy)
    total = tot["unified_total"]
    out: list[str] = []
    w = 58

    out.append("Neko per-rank memory model")
    out.append("=" * 74)
    out.append(f"  elements (nelv)      {cfg.nelv}")
    out.append(f"  space                lx = {cfg.lx}"
               + (f", lxd = {cfg.lxd}" if cfg.dealias else " (no dealiasing)"))
    out.append(f"  dofs (n)             {cfg.n:,}")
    out.append(f"  precision            --enable-real={cfg.real_type}"
               f" ({REAL_TYPES[cfg.real_type]['desc']}),"
               f" rp = {cfg.rp_bytes} B")
    out.append(f"  velocity             {cfg.vel_solver} + {cfg.vel_precon}"
               f", projection {cfg.vel_projection_dim}")
    out.append(f"  pressure             {cfg.prs_solver} + {cfg.prs_precon}"
               f", projection {cfg.prs_projection_dim}")
    if cfg.prs_precon == "phmg":
        out.append("  phmg levels          lx = "
                   + ", ".join(str(x) for x in cfg.phmg_levels)
                   + f"; TreeAMG coarse grid, {cfg.tamg_levels} levels")
    out.append(f"  statistics           {cfg.stats}")
    out.append("  zero-copy            "
               + ("on (NEKO_HIP_ZEROCOPY=1)" if cfg.zero_copy
                  else "off (replicated buffers)"))
    out.append("")

    if show_terms:
        out.append("Terms")
        out.append("-" * 74)
        group = None
        for t in sorted(steady(terms),
                        key=lambda t: (t.group, -t.total(cfg.zero_copy))):
            if t.group != group:
                group = t.group
                out.append(f"\n  [{group}]")
            b = t.total(cfg.zero_copy)
            flag = " " if t.exact else "~"
            out.append(f"  {flag}{t.name[:w - 1]:<{w}}{humanize(b):>12}"
                       f"{100 * b / total:>7.1f}%")
        out.append("")
        out.append("  ~ = rests on a modelling assumption, see --provenance")
        out.append("")

    out.append("Totals")
    out.append("-" * 74)
    by_group: dict[str, int] = {}
    for t in steady(terms):
        by_group[t.group] = by_group.get(t.group, 0) + t.total(cfg.zero_copy)
    for g, b in sorted(by_group.items(), key=lambda kv: -kv[1]):
        out.append(f"   {g:<{w}}{humanize(b):>12}{100 * b / total:>7.1f}%")
    out.append("-" * 74)
    out.append(f"   {'device-mapped arrays (host + device)':<{w}}"
               f"{humanize(tot['mapped']):>12}")
    out.append(f"   {'host-only allocations':<{w}}"
               f"{humanize(tot['host_only']):>12}")
    out.append(f"   {'device-only allocations':<{w}}"
               f"{humanize(tot['device_only']):>12}")
    out.append("")
    out.append(f"   {'TOTAL, unified memory (APU)':<{w}}{humanize(total):>12}")

    other = totals(terms, not cfg.zero_copy)["unified_total"]
    if cfg.zero_copy:
        out.append(f"   {'  same case with zero-copy off':<{w}}"
                   f"{humanize(other):>12}"
                   f"   ({other / total:.2f}x)")
    else:
        out.append(f"   {'  same case with zero-copy on':<{w}}"
                   f"{humanize(other):>12}"
                   f"   ({other / total:.2f}x)")
    out.append("")
    out.append(f"   {'bytes per dof':<{w}}{total / cfg.n:>11.1f} B")
    out.append(f"   {'bytes per element':<{w}}{total / cfg.nelv:>11.0f} B")
    field = cfg.n * cfg.rp_bytes
    out.append(f"   {'in units of one solution field (n * rp)':<{w}}"
               f"{total / field:>11.1f}")

    peak, driver = setup_peak(terms, cfg.zero_copy)
    out.append("")
    out.append("Setup")
    out.append("-" * 74)
    out.append(f"   {'peak during initialisation':<{w}}{humanize(peak):>12}"
               f"   ({peak / total:.2f}x)")
    out.append(f"   driven by: {driver}")
    transients = sorted((t for t in terms if t.transient),
                        key=lambda t: -t.total(cfg.zero_copy))
    for t in transients[:5]:
        b = t.total(cfg.zero_copy)
        if b > 0.01 * total:
            out.append(f"     {t.name[:w - 3]:<{w - 2}}{humanize(b):>12}")

    if budget is not None:
        out.append("")
        out.append("Budget")
        out.append("-" * 74)
        out.append(f"   {'available per rank':<{w}}{humanize(budget):>12}")
        out.append(f"   {'headroom (steady state)':<{w}}"
                   f"{humanize(budget - total):>12}")
        out.append(f"   {'headroom (setup peak)':<{w}}"
                   f"{humanize(budget - peak):>12}")
        fit = fit_elements(cfg, budget)
        out.append(f"   {'max elements per rank at this lx':<{w}}{fit:>12,}")
        out.append(f"   {'max dofs per rank':<{w}}{fit * cfg.lx ** 3:>12,}")
        alt = Config(**{**cfg.__dict__, "zero_copy": not cfg.zero_copy})
        alt_fit = fit_elements(alt, budget)
        label = ("without zero-copy" if cfg.zero_copy else "with zero-copy")
        out.append(f"   {'max elements ' + label:<{w}}{alt_fit:>12,}")
        if alt_fit:
            out.append(f"   {'capacity ratio':<{w}}{fit / alt_fit:>11.2f}x")

    return "\n".join(out)


def fit_elements(cfg: Config, budget: int) -> int:
    """Largest ``nelv`` whose modelled footprint fits in ``budget`` bytes."""

    def footprint(nelv: int) -> int:
        probe = Config(**{**cfg.__dict__, "nelv": nelv})
        return totals(build_terms(probe), probe.zero_copy)["unified_total"]

    if footprint(1) > budget:
        return 0
    lo, hi = 1, 2
    while footprint(hi) <= budget:
        lo, hi = hi, hi * 2
        if hi > 10 ** 12:
            break
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if footprint(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def provenance(terms: list[Term]) -> str:
    out = ["Term provenance", "=" * 74]
    group = None
    for t in sorted(terms, key=lambda t: (t.group, t.name)):
        if t.group != group:
            group = t.group
            out.append(f"\n[{group}]")
        out.append(f"  {t.name}"
                   + ("   [freed after setup]" if t.transient else ""))
        out.append(f"      phase  {PHASES[t.phase]}")
        out.append(f"      size   {t.formula}"
                   + ("" if t.exact else "   (modelling assumption)"))
        out.append(f"      source {t.source}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def selftest() -> int:
    """Internal consistency checks.  Returns the number of failures."""
    failures = 0

    def check(cond: bool, what: str) -> None:
        nonlocal failures
        if not cond:
            failures += 1
            print(f"FAIL: {what}", file=sys.stderr)

    cfg = Config(nelv=8000, lx=8)
    terms = build_terms(cfg)
    zc = totals(terms, True)
    rep = totals(terms, False)

    check(rep["unified_total"] - zc["unified_total"] == zc["mapped"],
          "replicated total exceeds zero-copy total by the mapped bytes")
    check(zc["unified_total"]
          == zc["host_only"] + zc["device_only"] + zc["mapped"],
          "zero-copy total is host + device + mapped")
    check(all(t.mapped >= 0 and t.host >= 0 and t.device >= 0 for t in terms),
          "no negative term")
    check(all(t.source for t in terms), "every term cites a source")

    # Linearity in nelv: every modelled allocation is proportional to nelv.
    a = totals(build_terms(Config(nelv=1000, lx=8)), True)["unified_total"]
    b = totals(build_terms(Config(nelv=2000, lx=8)), True)["unified_total"]
    check(abs(b - 2 * a) < 0.02 * b, "footprint is linear in nelv")

    # gs entries: lx^3 - (lx-2)^3 per element, and every point of an lx=2
    # element is a surface point.
    check(Config(nelv=1, lx=8).gs_entries() == 8**3 - 6**3, "gs entry count")
    check(Config(nelv=1, lx=2).gs_entries() == 8, "gs entry count at lx=2")

    # Every configure --enable-real value is modelled, and the footprint is
    # monotone in rp with the fixed-width arrays as the only offset.
    by_type = {rt: totals(build_terms(Config(nelv=8000, lx=8, real_type=rt)),
                          True)["unified_total"] for rt in REAL_TYPES}
    check(by_type["ssp"] < by_type["sp"],
          "ssp is smaller than sp: the xp reduction buffer is narrower")
    check((by_type["sp"] - by_type["ssp"]) < 1e-5 * by_type["sp"],
          "...but only the reduction buffer differs, so by under 0.001%")
    check(by_type["sp"] < by_type["dp"] < by_type["qp"],
          "the footprint grows with rp")
    check(0.50 < by_type["sp"] / by_type["dp"] < 0.55,
          "single precision roughly halves the footprint")
    check(1.95 < by_type["qp"] / by_type["dp"] < 2.0,
          "quad precision roughly doubles the footprint")
    # Affine in rp: T(rp) = A*rp + B, with B the fixed-width arrays. Fitted
    # on ssp/dp, where xp tracks rp, and checked against qp.
    a = (by_type["dp"] - by_type["ssp"]) / 4
    b = by_type["dp"] - 8 * a
    check(abs(16 * a + b - by_type["qp"]) < 1e-6 * by_type["qp"],
          "the footprint is affine in rp")
    check(0 < b < 0.03 * by_type["dp"],
          "the rp-independent part is a small positive offset")
    try:
        Config(real_type="fp16")
        check(False, "an unknown --enable-real value is rejected")
    except ValueError:
        pass

    # fit_elements inverts the footprint.
    budget = 64 * 2**30
    fit = fit_elements(cfg, budget)
    f_ok = totals(build_terms(Config(**{**cfg.__dict__, "nelv": fit})),
                  True)["unified_total"]
    f_over = totals(build_terms(Config(**{**cfg.__dict__, "nelv": fit + 1})),
                    True)["unified_total"]
    check(f_ok <= budget < f_over, "fit_elements finds the largest nelv")

    # Turning features off only ever removes memory.
    base = totals(build_terms(cfg), True)["unified_total"]
    for name, override in (("dealias", {"dealias": False}),
                           ("stats", {"stats": "none"}),
                           ("phmg", {"prs_precon": "jacobi"})):
        less = totals(build_terms(Config(**{**cfg.__dict__, **override})),
                      True)["unified_total"]
        check(less < base, f"disabling {name} lowers the footprint")

    # Adding projection only ever adds memory.
    for name, override in (("pressure projection", {"prs_projection_dim": 20}),
                           ("velocity projection", {"vel_projection_dim": 20}),
                           ("flow rate forcing", {"flow_rate_force": True})):
        more = totals(build_terms(Config(**{**cfg.__dict__, **override})),
                      True)["unified_total"]
        check(more > base, f"enabling {name} raises the footprint")

    # Transients are excluded from the steady state but counted in the peak.
    check(any(t.transient for t in terms), "the model has transient terms")
    check(all(not t.transient for t in steady(terms)),
          "steady() drops the transients")
    peak, _ = setup_peak(terms, True)
    check(peak >= base, "the setup peak is at least the steady state")
    check(peak <= base + max(t.total(True) for t in terms if t.transient),
          "the setup peak adds at most one transient to the steady state")
    check(all(0 <= t.phase < len(PHASES) for t in terms),
          "phases are in range")

    # A transient large enough to dominate does show up as the peak driver.
    fake = build_terms(cfg) + [Term("test", "huge transient", host=10 * base,
                                    phase=len(PHASES) - 1, transient=True)]
    p_peak, p_driver = setup_peak(fake, True)
    check(p_peak > base and "huge transient" in p_driver,
          "a dominating transient sets the peak")

    # A cube-shaped subdomain shares less as it grows.
    small = Config(nelv=512, lx=8)
    large = Config(nelv=64000, lx=8)
    check((small.gs_shared() / small.gs_entries())
          > (large.gs_shared() / large.gs_entries()),
          "the shared fraction falls as the subdomain grows")
    forced = Config(nelv=8000, lx=8, shared_fraction=0.25)
    check(abs(forced.gs_shared() / forced.gs_entries() - 0.25) < 1e-6,
          "--shared-fraction overrides the geometric estimate")

    # The CLI runs end to end.
    with contextlib.redirect_stdout(io.StringIO()) as sink:
        rc = [main(["-e", "100", "--lx", "6", "--budget", "8GiB", "--terms",
                    "--provenance"]),
              main(["-e", "100", "--lx", "6", "--json"]),
              main(["--fit", "--budget", "8GiB", "--lx", "6"])]
    check(rc == [0, 0, 0], "the CLI runs end to end")
    check(sink.getvalue().count("Neko per-rank memory model") == 2,
          "the CLI prints a report")

    if failures == 0:
        print("selftest: all checks passed")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-e", "--elements", type=int, default=8000,
                   help="elements on this rank (default: 8000)")
    p.add_argument("--elements-total", type=int,
                   help="elements in the whole mesh; divided by --ranks")
    p.add_argument("--ranks", type=int, default=1,
                   help="MPI ranks, used with --elements-total (default: 1)")
    p.add_argument("--lx", type=int, default=8,
                   help="points per direction, order + 1 (default: 8)")
    p.add_argument("--lxd", type=int,
                   help="dealiasing points per direction (default: (3*lx)/2)")
    p.add_argument("--real-type", default="dp", choices=list(REAL_TYPES),
                   help="working precision, as passed to Neko's "
                        "configure --enable-real (default: dp)")
    p.add_argument("--single-precision", dest="real_type",
                   action="store_const", const="sp",
                   help="shorthand for --real-type sp")

    p.add_argument("--no-zero-copy", action="store_true",
                   help="model replicated host/device buffers instead")

    p.add_argument("--no-dealias", action="store_true")
    p.add_argument("--velocity-solver", default="cg",
                   choices=sorted(k for k in KSP_WORK_VECTORS if k != "gmres"))
    p.add_argument("--velocity-preconditioner", default="jacobi",
                   choices=sorted(PC_WORK_VECTORS))
    p.add_argument("--pressure-solver", default="gmres",
                   choices=sorted(KSP_WORK_VECTORS))
    p.add_argument("--pressure-preconditioner", default="phmg",
                   choices=sorted(list(PC_WORK_VECTORS) + ["phmg"]))
    p.add_argument("--phmg-schedule", default=",".join(str(s) for s in
                                                       PHMG_DEFAULT_SCHEDULE),
                   help="comma-separated pcoarsening_schedule (default: 3,1)")
    p.add_argument("--tamg-levels", type=int, default=TAMG_DEFAULT_LEVELS)
    p.add_argument("--pressure-projection", type=int, default=0)
    p.add_argument("--velocity-projection", type=int, default=0)
    p.add_argument("--stats", default="full",
                   choices=["none", "basic", "full"])
    p.add_argument("--flow-rate-force", action="store_true",
                   help="case.fluid.flow_rate_force is set")
    p.add_argument("--scratch-fields", type=int, default=SCRATCH_PEAK_FIELDS)
    p.add_argument("--shared-fraction", type=float,
                   help="fraction of gs entries shared with another rank "
                        "(default: derived from a cube-shaped subdomain)")

    p.add_argument("--budget", type=str,
                   help="memory available per rank, e.g. 120GiB")
    p.add_argument("--fit", action="store_true",
                   help="report the largest nelv fitting in --budget")
    p.add_argument("--terms", action="store_true",
                   help="list every term, not just the group totals")
    p.add_argument("--provenance", action="store_true",
                   help="print each term's formula and source location")
    p.add_argument("--json", action="store_true",
                   help="machine-readable output")
    p.add_argument("--selftest", action="store_true")

    args = p.parse_args(argv)

    if args.selftest:
        return 1 if selftest() else 0

    nelv = args.elements
    if args.elements_total is not None:
        nelv = max(args.elements_total // max(args.ranks, 1), 1)

    budget = parse_size(args.budget) if args.budget else None

    cfg = Config(
        nelv=nelv,
        lx=args.lx,
        lxd=args.lxd,
        real_type=args.real_type,
        zero_copy=not args.no_zero_copy,
        dealias=not args.no_dealias,
        vel_solver=args.velocity_solver,
        vel_precon=args.velocity_preconditioner,
        prs_solver=args.pressure_solver,
        prs_precon=args.pressure_preconditioner,
        phmg_schedule=tuple(int(x)
                            for x in args.phmg_schedule.split(",") if x),
        tamg_levels=args.tamg_levels,
        prs_projection_dim=args.pressure_projection,
        vel_projection_dim=args.velocity_projection,
        stats=args.stats,
        flow_rate_force=args.flow_rate_force,
        scratch_fields=args.scratch_fields,
        shared_fraction=args.shared_fraction,
    )

    if args.fit:
        if budget is None:
            p.error("--fit requires --budget")
        cfg = Config(**{**cfg.__dict__, "nelv": fit_elements(cfg, budget)})

    terms = build_terms(cfg)

    if args.json:
        print(json.dumps({
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.__dict__.items()},
            "derived": {"n": cfg.n, "n_dealias": cfg.n_dealias,
                        "gs_entries": cfg.gs_entries(),
                        "gs_shared": cfg.gs_shared()},
            "totals": totals(terms, cfg.zero_copy),
            "setup_peak": setup_peak(terms, cfg.zero_copy)[0],
            "setup_peak_driver": setup_peak(terms, cfg.zero_copy)[1],
            "budget": budget,
            "max_elements": fit_elements(cfg, budget) if budget else None,
            "terms": [{"group": t.group, "name": t.name, "mapped": t.mapped,
                       "host": t.host, "device": t.device,
                       "total": t.total(cfg.zero_copy), "formula": t.formula,
                       "source": t.source, "exact": t.exact,
                       "phase": PHASES[t.phase], "transient": t.transient}
                      for t in terms],
        }, indent=2))
        return 0

    print(report(cfg, terms, budget, args.terms))
    if args.provenance:
        print()
        print(provenance(terms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
