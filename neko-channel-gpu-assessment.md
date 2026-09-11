# Neko on NVIDIA GPUs: the Pn-Pn pressure operator, and what a specialised turbulent-channel code could buy

**Scope.** Two connected questions about [Neko](https://github.com/ExtremeFLOW/neko), a spectral-element CFD code:

1. Is the Pn-Pn pressure operator symmetric positive definite, and is it the same operator you could solve globally? *(Measured. Section 1.)*
2. If you extracted Neko's kernels into a specialised turbulent-channel-flow code for NVIDIA GPUs, what could be optimised, and how much would it buy? *(Analysis. Section 2.)*

**Provenance.** Neko 1.99.9, commit `ce9b260`, September 2026. Section 1's numbers were produced on a CPU build (gfortran 13.3, OpenMPI, double precision, serial) with a test harness committed at `contrib/prs_operator_test/`. Section 2 is a static analysis of the source plus a bandwidth model; it contains **no GPU measurements** and is labelled accordingly throughout.

**Claim labelling.** Every quantitative statement is tagged:
`[M]` measured here · `[C]` read directly from the source, with file:line · `[E]` estimated from a model.

---

## Summary of findings

**On the pressure operator (Section 1):**

- On a channel with periodic streamwise/spanwise directions and no-slip walls, the operator Neko's Krylov solver applies is **symmetric to machine precision and positive semi-definite**, with a one-dimensional null space of constants. `[M]`
- The widespread belief that "the SEM pressure operator is not SPD" is **true of any configuration with a strong pressure boundary condition**, because Neko masks the operator output only — it applies `M·A`, not `M·A·M`. The asymmetry is 2.3e-2 relative, far above round-off. But it lives entirely in columns the Krylov iteration never excites. `[M]`
- On a **tensor-product** mesh the operator is **exactly** the separable Kronecker sum of one-dimensional SEM stiffness and mass operators — to 4e-14. Separability requires a tensor product, **not** uniform spacing: the test mesh is non-uniform in two of three directions. `[M]`
- A direct separable (fast-diagonalisation) solve reproduces Neko's GMRES answer and is **exact**: residual 8e-14 against GMRES's 1.3e-8 after 33 preconditioned iterations. `[M]`
- The velocity Helmholtz operator is separable too, and stays so under the no-slip wall mask. `[M]`

**On a specialised channel code (Section 2):**

- Neko's operator kernels are already heavily optimised (runtime autotuning, fp64 tensor cores, TMA staging) and are **bandwidth-bound**. The opportunity is in their *inputs*, not their arithmetic. `[C]`
- On an axis-aligned box mesh the six geometric factors collapse to **three scalars per wall-normal element layer**, and three of the six are identically zero. `[M, algebraically]`
- Neko already contains a switched-off prototype of exactly this compression. `[C]`
- A realistic band is **~2x** on time-to-solution for a structured-metric rewrite keeping the same algorithms, and **~2.5-4x** more elements per GPU depending on whether statistics are collected. A direct separable pressure solve could go substantially further but trades nearest-neighbour halo exchange for global transposes. `[E]`
- Several parts of the argument weaken or fail for an LES: a spatially varying eddy viscosity forces a different, non-separable velocity operator. `[C]`

---

# Section 1 — The Pn-Pn pressure operator

## 1.1 What the operator actually is

In Neko's `pnpn` scheme, the pressure residual routine sets the operator coefficients immediately before the solve `[C]`:

```fortran
! src/fluid/bcknd/cpu/pnpn_res_cpu.f90:74-78
c_Xh%h1(i,1,1,1) = 1.0_rp / rho_val
c_Xh%h2(i,1,1,1) = 0.0_rp
c_Xh%ifh2 = .false.
```

`h1` depends on a single scalar — not on the velocity, not on `dt`, not on `mu`. The pressure operator is therefore `(1/ρ)·L` where `L` is the SEM Laplacian, with a static geometry. Everything complicated in Pn-Pn (the curl-curl term, the surface/Neumann terms) lives in the **right-hand side**, not in the operator.

Inside the Krylov solver the operator is applied as this triple `[C]`, identically in GMRES (`src/krylov/bcknd/cpu/gmres.f90:225-228`), CG (`cg.f90:207-209`), and the projection basis (`projection.f90:351-353`):

```fortran
call Ax%compute(w, z, coef, msh, Xh)   ! element-local Helmholtz
call gs_h%op(w, n, GS_OP_ADD)          ! direct stiffness summation
call bc_projector%apply(w, n)          ! zero at strong boundaries
```

## 1.2 Method

`contrib/prs_operator_test/assemble.f90` runs a real `pnpn` case for several timesteps, re-executes `prs_res%compute` (the routine that defines the operator), then applies the triple above column by column to build the global matrix.

Three choices make the result meaningful rather than circular:

- **Dof identity comes from the operator itself.** The unique-dof numbering is obtained by setting `v(i) = i` and running a `GS_OP_MIN` gather-scatter, so it is the same equivalence relation the assembly step uses. It cannot merge dofs the operator treats as distinct, nor split identical ones.
- **The inner product is the solver's own.** Since `coef%mult = 1/multiplicity`, summing `u(rep(i))·Au(rep(i))` over unique dofs is *exactly* the `glsc3(·,·,coef%mult)` product the Krylov solver minimises in. "The matrix is symmetric" therefore means "the operator is self-adjoint in the norm the solver actually uses" — which is the statement that matters.
- **Assumptions are measured, not asserted.** The harness reports `max|w(i) − w(rep(l2g(i)))|` (representative-copy consistency: `bc_projector%apply` is a plain index list that propagates nothing, so this could fail on a mesh where a boundary zone masks a dof in one element but not its neighbour) and `sum_unique(b)` (right-hand-side consistency). Both come out at 0 and 3e-15 respectively for the cases below. `[M]`

Matrix-free checks (`⟨u,Av⟩` vs `⟨v,Au⟩`, `⟨v,Av⟩`, `A·1`) run at sizes too large to assemble densely.

**Test mesh.** Channel `2π × 2 × π`, periodic in x and z, no-slip walls in y. Deliberately **non-uniform in x** (element widths 1.65 / 2.65 / 1.98) and tanh-stretched in y (height ratio 8.1). Order 4 (`lx = 5`), 54 elements, 3600 unique dofs. Operator state read from the live objects: `h1 ≡ 1.0`, `h2 ≡ 0`, `ifh2 = F`, `prs_dirichlet = F`, pressure mask size 0.

## 1.3 Result: symmetric, positive semi-definite `[M]`

| quantity | value |
| --- | --- |
| `max\|A − Aᵀ\| / max\|A\|` | **1.25e-16** (machine eps = 2.22e-16) |
| `‖A − Aᵀ‖_F / ‖A‖_F` | 6.44e-17 |
| matrix-free `⟨u,Av⟩` vs `⟨v,Au⟩`, relative | 1.57e-16 |
| `‖A·1‖_∞` | 2.84e-14 |
| negative eigenvalues | **0** |
| zero eigenvalues | **1** (the constant) |
| 2nd smallest / largest eigenvalue | 9.33e-03 / 1.07e+02 |
| `max\|Im λ\| / max\|Re λ\|` (full nonsymmetric eigensolve) | 1.17e-17 |
| condition number over the non-null modes | 1.14e+04 |

Re-run matrix-free at order 7 with 96 elements (33,516 dofs): relative asymmetry 1.5e-16, `⟨v,Av⟩ > 0` on every sample, `‖A·1‖_∞ = 1.2e-14`. `[M]`

**The operator is SPD on the complement of the constants.** CG is admissible; the constant mode is what Neko's `ortho` removes.

## 1.4 Result: where the "not SPD" belief comes from — and it is correct there `[M]`

Re-run with x non-periodic, an inflow on one face and an `outflow` (Dirichlet pressure) on the other:

| quantity | value |
| --- | --- |
| `max\|A − Aᵀ\| / max\|A\|` | **2.26e-02** — not round-off |
| all-zero rows / all-zero columns | 204 / 0 |
| restricted to the 2448 unmasked dofs: symmetry | 1.54e-16 |
| restricted: min / max eigenvalue | 9.24e-04 / 3.21e+01 |
| restricted: negative / zero eigenvalues | 0 / 0 |
| `M·A·M` symmetry | 1.54e-16 |

`scalar_bc_projector%apply` zeroes the operator **output** only. Neko therefore hands the Krylov solver `M·A`, which is genuinely non-symmetric; the asymmetry sits entirely in the masked columns. Restricted to the unmasked dofs the operator is symmetric and **strictly** positive definite — the Dirichlet condition removes the constant null mode.

This is why GMRES rather than CG is the sensible default, and is almost certainly the origin of the folklore. Two qualifications:

- In principle the iteration never sees the asymmetry: the initial residual is masked (`fluid_pnpn.f90:857`) and every operator output is masked (`gmres.f90:248`), so all Krylov vectors lie in `{x : x_mask = 0}`, which is `A`-invariant. `[C]`
- But that depends on the preconditioner also preserving the mask, and the CPU HSMG V-cycle does **not** mask its output on exit (`pc_hsmg.f90:621-623`). **This was not tested here** — it would need `max|z(mask)|` measured over the real iterates.

## 1.5 Result: the operator is exactly separable `[M]`

Building the one-dimensional assembled SEM stiffness `K` and diagonal GLL mass `M` per direction from first principles and comparing against

> `A ≟ Kx⊗My⊗Mz + Mx⊗Ky⊗Mz + Mx⊗My⊗Kz`

| quantity | value |
| --- | --- |
| `max\|A − A_sep\| / max\|A\|` | **4.26e-14** |
| `‖A − A_sep‖_F / ‖A‖_F` | 3.20e-14 |

**Separability requires a tensor-product mesh, not a uniform one.** The test mesh is non-uniform in two of three directions and the identity still holds to round-off. Uniformity matters only for the *transform*, not the *structure* — see 1.7.

The mechanism: `coef_generate_geo` folds the quadrature weight into the metric (`src/sem/coef.f90:1130`), so for an axis-aligned box element `G12 = G13 = G23 ≡ 0` exactly and `G11 = s_e · w3(i,j,k)` with `s_e` one scalar per element. `[C]`

## 1.6 Result: a direct separable solve reproduces Neko's answer, exactly `[M]`

Fast diagonalisation (Lynch–Rice–Thomas): solve `K v = λ M v` per direction; the 3-D eigenvalues are `λx + λy + λz`; exactly one is zero (0+0+0 — the constant), and is dropped.

A smooth manufactured pressure field `p*`, right-hand side `b = A p*` formed through Neko's own operator, solved two ways:

| | residual `‖Ax − b‖/‖b‖` | error vs exact |
| --- | --- | --- |
| Neko GMRES + HSMG, 33 iterations | 1.31e-08 | 3.71e-09 |
| **direct fast diagonalisation** | **8.09e-14** | **8.05e-14** |
| agreement between the two | | 3.71e-09 |

The direct solver reproduces Neko's answer to the Krylov tolerance, and is exact rather than iterative.

**The operator is one fixed matrix for the whole run.** Assembled after 4 steps and after 9 steps it is bit-identical. `[M]` This is not independent evidence — it follows necessarily from `h1 = 1/ρ` with constant ρ and a static mesh — but it is the property that makes a one-off setup-time diagonalisation viable. It fails under ALE (`fluid_pnpn.f90:797-812` recomputes the metrics) and under the stress formulation, where `h1` is genuinely field-valued. `[C]`

## 1.7 Result: the velocity Helmholtz operator is separable too `[M]`

At the velocity solve, `h1 = μ` and `h2 = ρ·bd/dt`, both constant scalars `[C]`. Assembling the scalar Helmholtz that `Ax_vel%compute_vector` applies per component, with the real per-component no-slip mask (288 masked rows = exactly the two wall planes):

| quantity | value |
| --- | --- |
| `max\|A − Aᵀ\| / max\|A\|` | 1.19e-19 |
| min / max eigenvalue | 6.35e-02 / 1.46e+01 (condition number 230) |
| negative eigenvalues | 0 |
| `max\|A − (μ·Lap_sep + h2·Mass_sep)\| / max\|A\|` | **2.85e-15** |

Removing the two wall planes is a one-dimensional restriction in y, so the tensor structure survives the Dirichlet condition. Both solves in the timestep are therefore directly invertible on a channel mesh — subject to the caveats in 1.8.

*Note: this is the per-component operator. The full velocity solve additionally applies `rotate_cyc` (a no-op without cyclic boundary conditions) and solves the three components together.*

## 1.8 Result: is an FFT applicable? Only where the spacing is uniform `[M]`

GLL points are non-uniform *inside* an element, so a plain FFT over all points is not available. But on a **uniform periodic element line** the assembled one-dimensional operator is **block-circulant** with block size `lx−1`, so a DFT *across elements* block-diagonalises it:

| | off-block-diagonal mass after FFT across elements |
| --- | --- |
| uniform spacing | **1.23e-16** |
| non-uniform spacing (control) | 1.59e-01 |

The mass diagonal is identical across blocks (spread exactly 0), so the generalised eigenproblem decouples the same way. A periodic direction with uniform elements therefore costs an FFT of length `n_elem` plus a small dense `(lx−1)` solve per wavenumber, rather than a dense `N×N` transform. Channel meshes are uniform in x and z, so this applies; the wall-normal direction stays dense, and `n_y` is small.

## 1.9 What breaks these results

| condition | pressure operator | velocity operator |
| --- | --- | --- |
| non-tensor-product / curved mesh | separability lost (`G12,G13,G23 ≠ 0`) | lost |
| variable ρ | lost | lost |
| LES eddy viscosity (`nut_field`) | **survives** (operator only uses `1/ρ`) | **lost** — see below |
| SVV | survives | lost (`h1` becomes a full field, `spectral_vanishing_viscosity.f90:151`) |
| ALE / moving mesh | lost (metrics recomputed each step) | lost |
| Dirichlet pressure bc | still SPD on the unmasked subspace; `M·A` not symmetric | n/a |

The LES row is the sharpest practical limit `[C]`: setting `case.fluid.nut_field` without `full_stress_formulation` is a hard error (`fluid_pnpn.f90:311-316`), and the stress formulation switches `Ax_vel` to `ax_helm_full`, a coupled operator on the nine Jacobian metrics rather than the six `G_ij`. **A constant-`h1` velocity solve describes a DNS, not a wall-modelled LES.**

## 1.10 Reproducing

`contrib/prs_operator_test/` — `./mkmesh.sh`, then `./assemble channel.case`, then the Python analysis scripts. Serial, CPU, double precision only; the program refuses to run otherwise. The README carries the full result tables.

---

# Section 2 — A specialised channel code: what could be optimised

**Everything in this section is static analysis and modelling, not measurement.** The target configuration assumed throughout: turbulent channel, `pnpn`, `lx = 8` (order 7), dealiasing on (`lxd = 12`), fp64, no solution projection, p-multigrid pressure preconditioner, Jacobi + coupled CG velocity.

## 2.1 The starting point: the operators are already good, and already bandwidth-bound

Neko's own documentation quantifies this `[C]`:

> "at `lx = 8` an element is nine cubes of 4 kB, of which the seven geometric factors alone are 78% of the read traffic" — `doc/pages/user-guide/performance.md:188`
>
> "`Ax` is strongly bandwidth bound, roughly 1.6 flop/byte at `lx = 8` against a ridge point near 9 on GH200" — `performance.md:298`

*A precision note on the first quote:* the output cube `w` is write-only, so only eight of the nine cubes are read. The seven factors are 7/8 = 87.5% of **read** traffic; 78% is 7/9, their share of **total** traffic. The doc's own table (`performance.md:252-258`) lists the scalar `Ax` as 8 batched copies, consistent with eight reads.

The kernels themselves are already autotuned at runtime across five formulations — 1-D, kstep, fp64 tensor-core (DMMA), Hopper TMA-staged, and mfma on HIP — with the winner cached per operator and polynomial order (`src/math/bcknd/device/cuda/ax_helm.cu:87-231`). `[C]` **Do not expect to beat these kernels by hand.** Their *inputs* are the opportunity.

## 2.2 The structured-grid collapse

For an axis-aligned tensor-product box mesh the metric is exactly:

- `G12 = G13 = G23 ≡ 0`
- `G11(i,j,k,e) = w3(i,j,k) · s_e` — one scalar per element, and since every element in a wall-normal layer is identical, **one scalar per y-layer**
- `jacinv`, `B`, `Binv`, `h1`, `h2` likewise; `mult` is separable into a length-`lx` array; the dof coordinates become three 1-D arrays

So `Ax` reads one cube (`u`) and writes one, instead of reading eight. That is a 4x cut in traffic and moves the arithmetic intensity from 1.6 to ~7 flop/byte — close to, and therefore **bounded by**, the ~9 ridge. The Ax-level gain saturates around 3-4x and cannot be extrapolated further.

**Neko already contains a prototype of this.** `coef_generate_geo_compressed` (`src/sem/coef.f90:1402`) deduplicates identical `G` tensors and builds a `compression_inds` lookup. The call is commented out at `coef.f90:445` and no kernel reads `Gij_compressed`. `[C]` For the reference channel (`examples/turb_channel`, 5832 elements, 18 wall-normal layers) it would compress 5832 → 18. Its matcher uses an absolute tolerance of `1e-7` and is quadratic in element count in the worst case, which is plausibly why it was parked. A related path is also scoped but unimplemented: `metric_sp_safe` / `NEKO_METRIC_COND_SP` (`coef.f90:71,119`) gate "a future reduced precision storage path".

**This is the cheapest available experiment.** Enabling the compression, replacing the matcher with a y-layer key, and adding a single `compression_inds[e]` indirection to the element base pointer in the existing kernels would put the compressed `G` set in L2 (18 layers × 8³ points × 6 arrays × 8 B ≈ 440 kB at `lx = 8`, against 40-50 MB of L2) and recover most of the read-traffic reduction — inside upstream Neko, with the autotuner and all backends intact. Measuring that first would calibrate everything else in this section.

## 2.3 The memory ledger `[C]`

In *field-equivalents* (1 FE = `lx³ · nelv · 8` bytes = 4096 B/element at `lx = 8`):

| block | FE | note |
| --- | ---: | --- |
| `coef_t`, fine level | 34.0 | 31 full-size arrays + 4 facet arrays |
| dofmap (device: x, y, z only) | 3.0 | `dof`/`shared_dof` are host-only |
| pnpn solution state | 36.0 | u,v,w,p, 6 lag, extrapolated, residuals, increments, forcing, AB/BDF history, properties |
| **dealiasing block** | **57.4** | 9 GL metrics + 8 GL work arrays at `(12/8)³ = 3.375x` |
| **GMRES(30) workspace** | **62.0** | `z(n,30)` + `v(n,30)` + w + r |
| velocity coupled fused CG | 41.0 | 10 + 3×10 p-space, + Jacobi diagonal |
| p-multigrid hierarchy | 9.8 | per-level `COEF_OPERATOR` coef + vectors + Chebyshev work |
| gather-scatter | 2.1 | index arrays: 16 B per gs dof |
| **total** | **245.3** | **1962 B per GLL point → ~69,700 elements per 70 GB GPU** |

With BiCGStab instead of GMRES(30): 190.3 FE → ~89,800 elements (**+29%**). Note that GMRES(30) is a case-file choice, not a code limitation — `fused_cg`, `fused_coupled_cg`, `pipecg`, `cheby`, `bicgstab` all exist. `[C]`

**A production channel run holds more than this.** `fluid_stats` with `set_of_stats: "full"` keeps **58 resident fields** (44 statistics + 9 gradients + 5 work, `src/fluid/fluid_stats.f90:192,201-227`) `[C]`. That is the single largest block in a real run, it is the scientific output the run exists to produce, and **no optimisation in this document removes it.**

The two largest reducible blocks are the dealiasing workspace and the Krylov workspace — not the geometry. Geometry is 34/245 ≈ 14% of the ledger (or 65/245 ≈ 27% counting the GL metrics).

## 2.4 Opportunities, ranked

**1. Collapse the geometric factors.** As in 2.2. Affects `Ax`, `opgrad`, `dudxyz`, `cdtp`, `conv1`. `[E]` ~3-4x on those kernels, bounded by the ridge.

**2. Fuse the dealiased convection.** `adv_dealias.f90` carries a source comment calling its device path *"extremely primitive and unoptimized"* (`:253`) `[C]`. Per timestep it does three GLL→GL interpolations, then per velocity component an `opgrad` at the dealiased order, a `vdot3`, a GL→GLL map and a `sub2` — with 17 arrays of persistent GL-order storage. `[C]` A fused kernel eliminates nearly all of that traffic and 57 FE of memory. Neko already has the cheaper convecting-field-in-`rst` formulation (`set_convect_rst`, `conv1`) but wires it only into the OIFS path. `[C]`

**3. Improve the curl-curl in the pressure residual — but not by fusing it.** Each `curl` is six separate `dudxyz` calls plus `sub3`s, which can become one kernel per curl. **But `curl` ends with a full gather-scatter** — `opcolv(B) → gs_h%op(ADD) → opcolv(Binv)` (`src/math/bcknd/cpu/opr_cpu.f90:173-178`) — so the two curls **cannot** be fused into a single kernel. `[C]` *(An earlier version of this analysis claimed they could; that was wrong.)*

**4. Direct-address the gather-scatter.** On a lexicographic structured grid the index arrays become computed offsets. Note Neko already detects contiguous runs (`gs_find_blks`) and already fuses the three velocity components into one halo round (`gs_op_vector3`) `[C]`, so this is "remove the residual index loads", not "replace the indirection".

**5. Drop the Krylov workspace.** Configuration, not engineering: BiCGStab (7 FE) or CG (4 FE) instead of GMRES(30) (62 FE). Since the operator is SPD (Section 1), CG is admissible on a channel provided the preconditioner is symmetric.

**6. Mixed precision.** The `rp` kind is fixed at build time (`src/config/num_types.f90.in`), and the default `--enable-real=dp` build sets the extended-accumulation kind `xp` equal to `rp`, so it is uniformly fp64. `[C]` There is **no** path for an fp32 preconditioner inside an fp64 solve — the standard nekRS trick. That is a genuine gap, worth perhaps 1.2-1.3x overall if the V-cycle dominates.

**7. Reduce host synchronisation in the Krylov loop.** `fusedcg_device_solve` performs **three** host-returning global reductions per iteration (two `device_glsc3` plus `device_fusedcg_part2`, each with its own `MPI_Allreduce` on a non-device-MPI build). `[C]`

## 2.5 The coarse-grid solve: the strongest structural argument

`tamg_device_matvec_flat_impl` (`src/multigrid/tree_amg.f90:586`) implements a *coarse*-level matvec by mapping the coarse vector back to the **finest** level, applying the full `Ax` there plus two gather-scatters and two masked gather/reduction passes, then mapping back. `[C]` The AMG hierarchy reduces vector length but **not work per matvec**; there is no assembled coarse matrix anywhere in `tree_amg`. With the defaults (3 AMG levels, Chebyshev degree 4) one coarse solve issues on the order of 20 such matvecs, each several kernel launches — a long tail of tiny, launch-latency-bound work.

Section 1 shows the structured channel makes this removable: the coarse operator is separable, so an **exact** direct coarse solve is available.

**One important qualification.** How much this matters depends on the preconditioner:

- With **p-multigrid (`phmg`)** the V-cycle is `Ax`-based — approximately six `Ax` applications per level per cycle at the default `smoother_iterations = 3` (the down-leg pre-smooth costs two, not three, because its first Chebyshev iteration runs with a zero initial guess) `[C]`. Here both the geometry collapse and the coarse-solve replacement pay directly.
- With **HSMG** (Schwarz + FDM, the default in Neko's own channel example) the V-cycle contains **no `Ax` application** outside the coarse solve — it is roughly ten halo exchanges, two overlapping Schwarz solves and four interpolations `[C]`. Collapsing `G` speeds up only the single `Ax` per Krylov iteration. HSMG also carries its own per-element geometry the ledger above does not count: FDM's `d(nl³, nelv)` and `s` with `nl = lx+2` (`src/math/fdm.f90:125-126`) `[C]`. And `pc_hsmg.f90:267` calls `msh%all_deformed()`, which globally marks every element deformed and disables the only axis-aligned fast path in the code — which exists on the CPU backends only, never in CUDA. `[C]`

## 2.6 A direct separable pressure solve: the large prize, and its price

Section 1 establishes the operator is exactly separable and constant in time. A fast-diagonalisation solve is therefore **exact** and needs no Krylov iteration at all.

Cost, with FFTs in the two uniform periodic directions and a dense transform in the wall-normal direction: on the order of a few `Ax`-equivalents of dense GEMM per solve, against O(100) `Ax`-equivalents for a preconditioned Krylov solve. `[E]`

**The price is the communication pattern, and it is the single biggest risk to any speedup estimate.** A direct solve needs a global transform in each direction, i.e. pencil transposes (all-to-all) per solve, replacing Neko's nearest-neighbour halo exchange. That scales like a classical spectral channel code, not like Neko. On one to a few GPUs it is free; at scale it is the dominant question and **has not been measured here**.

Two further practical obstacles `[C]`: the operator is singular (pure Neumann; null space removed by `ortho`), and constant-mass-flux forcing performs a **second full pressure solve per timestep** (`src/fluid/fluid_volflow.f90:242`) which would have to go through the direct path too.

## 2.7 Costs a naive model ignores

A real `pnpn` timestep contains substantial work that is neither `Ax` nor gather-scatter `[C]`:

- **Constant-mass-flux forcing** (`flow_rate_force`, which every channel at fixed `Re_b` needs) runs a full extra pressure solve *and* a full extra coupled velocity solve on recompute (`fluid_volflow.f90:242,298`).
- **CFL is computed twice per step** with variable timestepping (`simulation.f90:135,137`), each a 13-array full-field pass plus an `MPI_Allreduce`.
- **Two global reductions per step** for the pure-Neumann pressure (`ortho` at `fluid_pnpn.f90:848,891`), plus **two unconditional global barriers** in the output controller even when nothing is written (`output_controller.f90:247,296`).
- **Five gather-scatter rounds in residual assembly alone**, before any Krylov iteration.
- **Simulation components run every step by default** — in the reference channel case `lambda2` (a full 9-component gradient tensor plus a 3×3 eigenvalue solve per point) has no `compute_control` and therefore runs every step.
- Source terms, weak boundary conditions, EXT/BDF assembly, lag rotation, material properties.

## 2.8 Estimates, with error bars

These are modelled, not measured `[E]`:

| | estimate | confidence |
| --- | --- | --- |
| `Ax`-class kernels, structured metrics | 3-4x | good — anchored to the documented roofline |
| dealiased convection, fused | 5-10x on that term | moderate |
| **time-to-solution, structured rewrite, same algorithms** | **~2x** | **low-moderate** |
| plus fp32 preconditioner | ~1.2-1.3x further | moderate |
| plus direct separable pressure solve, 1-4 GPUs | substantially more; not bounded here | low — transposes unmeasured |
| elements/GPU, no statistics collected | ~3.5-4x | good — arithmetic from the ledger |
| **elements/GPU, with `fluid_stats` "full" resident** | **~2.5x** | **good** |

The memory numbers are arithmetic and are reliable. **The time-to-solution numbers are not measured and should be treated as a hypothesis to test, not a result.** The honest way to calibrate them is 2.2: enable the existing compression in upstream Neko and measure.

## 2.9 Effort

The CUDA layer (~21k lines) lifts cleanly — `ax_helm_kernel.h` needs only `elem_block.h`, the DMMA headers and `device_config.h`. The Fortran layer (~111k lines in the core directories) is where the abstraction lives, and almost none of it is wanted; expect to write 5-8k lines of new driver.

- Correct structured `pnpn` channel solver on one GPU, reusing Neko's kernels: **~2-3 months**
- Structured metrics, fused convection, direct-addressed gather-scatter, fp32 preconditioner, multi-GPU, validated against Neko: **~9-12 months**
- Direct separable pressure solve: **+2-3 months**

---

## Open questions

1. **Does the geometry collapse actually deliver 3-4x on `Ax` in situ?** Testable cheaply inside Neko (2.2). Everything downstream depends on it.
2. **Is the Krylov subspace really mask-invariant under every preconditioner?** Section 1.4 argues it is under Jacobi and (mostly) HSMG, but `pc_hsmg.f90:621-623` does not mask on exit. Measure `max|z(mask)|` over real iterates.
3. **What do the pencil transposes cost at the target scale?** The decisive unknown for the direct-solve path.
4. **Where does the time actually go per timestep on a GPU?** No profile informed this document. Section 2.7 lists costs a bandwidth model ignores; some may dominate.
5. **Is any of this compatible with the intended physics?** If the target is a wall-modelled LES rather than a DNS, the velocity operator is not separable and `h1` is not constant (1.9).

## Key source references

| topic | location |
| --- | --- |
| pressure operator definition | `src/fluid/bcknd/cpu/pnpn_res_cpu.f90:74-78` |
| operator application in the solver | `src/krylov/bcknd/cpu/gmres.f90:225-228` |
| geometric-factor compression (disabled) | `src/sem/coef.f90:445, 1402` |
| roofline / autotuner documentation | `doc/pages/user-guide/performance.md:113-300` |
| dealiased advection | `src/fluid/bcknd/advection/adv_dealias.f90:107-253` |
| matrix-free AMG coarse matvec | `src/multigrid/tree_amg.f90:586` |
| p-multigrid V-cycle and defaults | `src/multigrid/phmg.f90:134-186, 616` |
| GMRES workspace | `src/krylov/bcknd/device/gmres_device.F90:65, 174-175` |
| gather-scatter indirection | `src/gs/bcknd/device/cuda/gs_kernels.h:43-66` |
| precision kinds | `src/config/num_types.f90.in` |
| LES forces the stress formulation | `src/fluid/fluid_pnpn.f90:311-316` |
| test harness | `contrib/prs_operator_test/` |
