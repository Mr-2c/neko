# Memory model {#memory-model}

\tableofcontents

The performance guide (\ref performance) advises filling each device with as
many elements as will fit. This page says how many that is: it models the
per-rank memory footprint of a Neko case from the allocations the code
actually makes, so the element count can be chosen before the job is
submitted rather than found by watching runs die.

The model matters most on unified-memory APUs such as the AMD Instinct
MI300A, where host and device draw on one physical pool and the zero-copy
mapping option (`NEKO_HIP_ZEROCOPY=1`, see \ref installation) roughly halves
the footprint. Everything below applies to discrete GPUs too, with the host
and device halves read off separately.

A tool implementing the model ships in
`contrib/neko_memory_model/neko_memory_model.py`.

## How Neko allocates {#memory-model-allocation}

Neko allocates in three ways, and they behave differently under zero-copy:

| Kind | What it is | Zero-copy off | Zero-copy on |
| ---- | ---------- | ------------- | ------------ |
| `allocate` only | A host array never handed to the device: mesh connectivity, the dofmap's global ids, MPI bookkeeping | 1x | 1x |
| `device_map` | A host array plus a device pointer. Nearly the whole footprint | 2x: the Fortran array plus a `hipMalloc` replica | 1x: the device pointer aliases the host allocation |
| `device_alloc` | A device buffer with no host array: the gather-scatter exchange buffers | 1x | 1x |

On an APU there is one physical pool, so what has to fit is

\f[
  M = H + D + \begin{cases} M_\mathrm{mapped} & \text{zero-copy on} \\
                            2\,M_\mathrm{mapped} & \text{zero-copy off} \end{cases}
\f]

where \f$H\f$ is the host-only total, \f$D\f$ the device-only total and
\f$M_\mathrm{mapped}\f$ the mapped total. Because mapped arrays are 98% of
the footprint, zero-copy is very close to a factor of two in capacity.

On a discrete GPU the same three numbers split instead as
\f$M_\mathrm{mapped} + D\f$ on the device and \f$M_\mathrm{mapped} + H\f$ on
the host.

## Notation {#memory-model-notation}

With `nelv` elements on the rank and `lx` points per direction:

| Symbol | Meaning | Value |
| ------ | ------- | ----- |
| \f$n\f$ | local degrees of freedom | `nelv * lx^3` |
| \f$n_d\f$ | dealiasing-space dofs | `nelv * lxd^3`, `lxd = (3*lx)/2` by default |
| \f$n_f\f$ | facet points | `nelv * lx^2 * 6` |
| \f$m\f$ | gather-scatter entries | `nelv * (lx^3 - (lx-2)^3)`, the non-interior points |
| \f$m_s\f$ | of those, shared with another rank | `~ 6 * nelv^(2/3) * lx^2` for a cube-shaped subdomain |
| \f$m_l\f$ | of those, local to the rank | \f$m - m_s\f$ |
| `rp` | bytes per working real | set by `--enable-real`, see below |
| `xp` | bytes per extended real | set by `--enable-real`; only ever backs the reduction buffers at problem scale |

The natural unit is one solution field, \f$n \cdot\f$ `rp` — 31.25 MiB at
`lx = 8` with 8000 elements in double precision. Most terms are a whole
number of those.

## The terms {#memory-model-terms}

The table below is for the configuration this model was first written for:
dealiasing on, DNS (no LES or SVV), CG + Jacobi for velocity, GMRES + PHMG
for pressure with TreeAMG as PHMG's coarse-grid solver, no solution
projection on either solve, and `fluid_stats` with `set_of_stats = full`
written as a full 3D field (`avg_direction = none`).

| Term | Size | Multiples of \f$n\,\mathrm{rp}\f$ at `lx = 8` | Source |
| ---- | ---- | --- | ------ |
| GMRES work vectors | \f$(2 + 2 \cdot 30)\,n\,\mathrm{rp}\f$ | 62.0 | `gmres_device.F90:162-189` |
| `fluid_stats`, full set | \f$58\,n\,\mathrm{rp}\f$ | 58.0 | `fluid_stats.f90:201-266` |
| Dealiased advection | \f$(8 + 9)\,n_d\,\mathrm{rp}\f$ | 57.4 | `adv_dealias.f90:135-153`, `coef.f90:250-292` |
| Fluid fields | \f$36\,n\,\mathrm{rp}\f$ | 36.0 | `fluid_scheme_incompressible.f90`, `fluid_pnpn.f90` |
| Coefficients, `COEF_FULL` | \f$31\,n\,\mathrm{rp} + 4\,n_f\,\mathrm{rp}\f$ | 34.0 | `coef.f90:327-378` |
| PHMG + TreeAMG | \f$7\,n\,\mathrm{rp}\f$ on the fine level, then \f$20\,n_i\,\mathrm{rp}\f$ per coarse level plus its own dofmap and gather-scatter | 9.6 | `phmg.f90`, `tree_amg*.f90` |
| Scratch registry | \f$8\,n\,\mathrm{rp}\f$ | 8.0 | `pnpn_res_device.F90:303-310` |
| Dofmap | \f$3\,n\,\mathrm{rp} + 12n\f$ B | 5.1 | `dofmap.f90:117-151` |
| CG + Jacobi | \f$(4 + 1)\,n\,\mathrm{rp}\f$ | 5.0 | `cg_device.f90:86-94`, `pc_jacobi_device.F90:121` |
| Gather-scatter | \f$m_l(\mathrm{rp}{+}8) + m_s(4\,\mathrm{rp}{+}8)\f$ | 2.1 | `gather_scatter.f90`, `gs_device*.F90` |
| fld write staging | \f$3n \cdot 4\f$ B, or \f$3n \cdot 8\f$ at `dp` output precision | 1.5 | `fld_file.f90:344-346` |
| Mesh connectivity | \f$\approx 300\f$ B per element + two hash tables | 0.4 | `mesh.f90:88-137` |
| Reduction buffers | \f$\lceil n/1024\rceil\f$ elements of `rp` and of `xp`, pinned host + device | 0.01 | `math.hip:765-771` |
| Boundary masks | \f$\propto\f$ boundary dofs | 0.1 | `bc.f90:512-517` |
| **Total** | | **279** | |

Some consequences worth reading off the table:

- **GMRES is the single largest term.** Its restart depth is fixed at 30
  (`gmres_device.F90:65`), not settable from the case file, so it holds 62
  vectors whatever the case — even when PHMG reduces it to a handful of
  iterations. A symmetric solver holds 4 (CG) to 10 (pipelined CG) instead,
  but PHMG's Chebyshev smoothing makes the preconditioner non-stationary,
  which is why GMRES is the usual choice here. Treat the 62 as the price of
  that robustness, not as slack.
- **Dealiasing costs more than the fields it exists for.** At the default
  `lxd = 3lx/2` the dealiasing space is \f$(3/2)^3 = 3.375\f$ times larger than
  the solution space, and 17 arrays live there — nine geometry arrays in
  `coef_GL` and eight work arrays. Lowering `dealiased_polynomial_order`
  scales this term as `lxd^3`.
- **Full statistics are the cost of one more solver.** 44 mean fields, five
  work fields and nine gradient fields. `set_of_stats = basic` is 16 instead
  of 58.
- **Projection was off here**, which is also the default. Turning it on costs
  \f$(2L{+}1)\,n\,\mathrm{rp}\f$ for the pressure and
  \f$3(2L{+}1)\,n\,\mathrm{rp}\f$ for the velocity, so setting
  `projection_space_size` to 20 on both solvers would add 41 + 123 = 164
  fields — more than half of everything else put together.

## What fits {#memory-model-capacity}

Per-dof cost is nearly flat in `lx`, which makes the capacity easy to
remember: with this configuration in double precision, **about 2.2 kB per dof
with zero-copy and 4.4 kB without**.

Elements and dofs per rank fitting in 100 GiB, the model's default
configuration, in double precision:

| `lx` | `lxd` | elements, zero-copy | dofs, zero-copy | elements, replicated | dofs, replicated |
| ---- | ----- | ------------------- | --------------- | -------------------- | ---------------- |
| 4  | 6  | 675,460 | 43.2 M | 342,586 | 21.9 M |
| 6  | 9  | 217,364 | 47.0 M | 109,644 | 23.7 M |
| 8  | 12 |  94,006 | 48.1 M |  47,325 | 24.2 M |
| 10 | 15 |  48,610 | 48.6 M |  24,458 | 24.5 M |
| 12 | 18 |  28,271 | 48.9 M |  14,222 | 24.6 M |

100 GiB is an illustrative budget, not a machine constant: substitute the
memory actually available to a rank, which on an APU is the physical pool
less the OS, the HIP runtime, MPI and the page cache, divided by the ranks
sharing it. Run the tool with `--budget` set to your own figure.

### Working precision {#memory-model-precision}

Neko's working precision is a compile-time choice, `configure --enable-real`,
and it is the single largest lever on the footprint after the case itself.
Four values are accepted (`configure.ac:22-28,195-253`):

| `--enable-real` | `rp` | `xp` | Footprint vs `dp` |
| --------------- | ---- | ---- | ----------------- |
| `ssp` | REAL32, 4 B | REAL32, 4 B | 0.508x |
| `sp`  | REAL32, 4 B | REAL64, 8 B | 0.508x |
| `dp`  | REAL64, 8 B | REAL64, 8 B | 1.000x (default) |
| `qp`  | REAL128, 16 B | REAL128, 16 B | 1.983x |

The footprint is affine in `rp`, \f$M = A\,\mathrm{rp} + B\f$, with \f$B\f$
— the integer index arrays, the mesh, the dofmap's global ids and the `dp`
point coordinates — only about 1.6% of the `dp` total. That is why halving
`rp` takes 50.8% rather than exactly 50%, and doubling it takes 1.98x rather
than 2x.

`ssp` and `sp` differ by 32 kB at four million dofs — 0.0007%. `xp` is the
extended real used for accumulating reductions, and the only allocation of
that type which scales with the problem is the device reduction buffer at
\f$\lceil n/1024\rceil\f$ elements
(`src/math/bcknd/device/hip/math.hip:769`). Everything else typed `xp` is
fixed-size: the CPU GMRES Hessenberg and Givens arrays, at `m_restart`
(`src/krylov/bcknd/cpu/gmres.f90:61-66`). So choose between `ssp` and `sp` on
numerical grounds, not memory ones.

\warning `qp` sets the device kernels' `real` typedef to `long double`
(`configure.ac:227-236`, `src/device/device_config.h.in`), which HIP and CUDA
do not support in device code. It is a CPU configuration; the `qp` row is
given for completeness.

#### Elements per rank at `lx = 8`, `lxd = 12`

| `--enable-real` | 100 GB, zero-copy | 100 GB, replicated | 100 GiB, zero-copy | 100 GiB, replicated | B/dof |
| --------------- | ----------------- | ------------------ | ------------------ | ------------------- | ----- |
| `ssp` | 172,297 | 87,300 | 185,034 | 93,746 | 1134 |
| `sp`  | 172,296 | 87,300 | 185,032 | 93,746 | 1134 |
| `dp`  |  87,541 | 44,072 |  94,006 | 47,325 | 2231 |
| `qp`  |  44,121 | 22,141 |  47,378 | 23,774 | 4427 |

Both budget columns are given because 100 GB (decimal) is 93.1 GiB — a 7%
difference, which is about 6,000 elements here.

## Initialisation peak {#memory-model-setup-peak}

Some allocations exist only during setup and are freed before the time loop,
so a case can die at startup having fitted comfortably afterwards. The model
tracks these against the setup phase they occur in, since a transient only
competes with what is already resident when it happens.

The largest is the gather-scatter shared-dof hash table
(`gather_scatter.f90:633`, freed at `:1618`), sized at one slot per dof and
costing about 136 B per slot — roughly 136 B per dof, or 6% of the steady
state. It is built before the coefficients, fields, solvers and statistics
exist, so with this configuration it never sets the peak; on a CPU-only run,
where the steady state is far smaller, it can. Next are the local-dof table,
the mesh's point table, and the coarse PHMG levels' coefficient arrays, which
are allocated as `COEF_FULL` and cut down to `COEF_OPERATOR` at the end of
`coef_init` (`coef.f90:807-940`).

## Using the tool {#memory-model-tool}

    contrib/neko_memory_model/neko_memory_model.py --elements 8000 --lx 8 --budget 100GiB

Useful flags:

| Flag | Effect |
| ---- | ------ |
| `--terms` | list every term rather than group totals |
| `--provenance` | print each term's formula and the source lines it was read from |
| `--fit --budget X` | report the largest element count fitting in `X` |
| `--no-zero-copy` | model replicated host/device buffers |
| `--real-type ssp\|sp\|dp\|qp` | working precision, matching `configure --enable-real` |
| `--elements-total N --ranks R` | divide a whole mesh across ranks |
| `--stats`, `--pressure-solver`, `--pressure-preconditioner`, `--pressure-projection`, ... | vary the case |
| `--json` | machine-readable output |
| `--selftest` | run the model's internal consistency checks |

## Accuracy and limits {#memory-model-limits}

The model is analytic: it enumerates allocations by reading the source, and
has not been calibrated against measured resident memory. Terms marked `~` in
`--terms` output rest on a modelling assumption rather than a literal
allocation size; `--provenance` states each one. The main ones are:

- **Hash tables.** Neko's `htable_t` stores two unlimited-polymorphic
  allocatables per slot, so a slot costs an inline descriptor plus two small
  heap chunks. 136 B per slot is a gfortran/glibc estimate; another compiler
  or allocator will differ. Table sizes also depend on when quadratic probing
  triggers a doubling.
- **The shared gather-scatter fraction.** Derived from a cube-shaped
  subdomain unless `--shared-fraction` says otherwise. A real partitioner
  gives less regular subdomains, and a rank on the domain boundary shares
  less. This affects a term worth under 1%.
- **Mesh connectivity per element**, which includes a polymorphic element
  descriptor whose size is compiler-dependent.
- **The scratch registry peak**, taken as the eight fields the pnpn pressure
  residual holds at once. A user file or simulation component that requests
  more raises it.

Not modelled at all: the HIP runtime's own allocations and page tables, MPI
internal buffers, the mesh file read, restart-from-checkpoint (which holds a
second mesh and function space), scalars, LES and SVV, ALE (which allocates a
second coefficient set and a mesh-velocity solve), OIFS time interpolation,
and any user-file fields. Add headroom accordingly, and treat the numbers as
a planning tool rather than a guarantee.
