# Neko on NVIDIA GPUs: the Pn-Pn pressure operator, and what a specialised turbulent-channel code could buy

Two connected questions about [Neko](https://github.com/ExtremeFLOW/neko), a spectral-element CFD code:

1. Is the Pn-Pn pressure operator symmetric positive definite, and is it the same operator you could solve globally? — **measured**, Part 1.
2. If you extracted Neko's kernels into a specialised turbulent-channel code for NVIDIA GPUs, what could be optimised and how much would it buy? — **analysis**, Part 2.

**Provenance.** Neko 1.99.9, based on upstream commit `ce9b260` (September 2026). Part 1's numbers were produced on a CPU build (gfortran 13.3, OpenMPI, double precision, serial) using a test harness added for this work at `contrib/prs_operator_test/` — it is not in upstream Neko. Part 2 is a static analysis of the source plus a bandwidth model; it contains **no GPU measurements** and is labelled accordingly throughout.

**Claim labelling.** Every quantitative statement is tagged:

| tag | meaning |
| --- | --- |
| `[M]` | measured here, by the committed harness |
| `[C]` | read directly from the source, cited by file:line |
| `[E]` | estimated from a model; not measured |

**Terminology.** *SEM* — spectral element method. *GLL* — Gauss–Lobatto–Legendre, the quadrature/interpolation points inside each element; *GL* — Gauss–Legendre, the denser set used for dealiasing. *`Ax`* — Neko's name for the element-local Helmholtz/Laplacian operator application, the inner kernel of every solve. *DSS* — direct stiffness summation, the gather-scatter that makes an element-local result continuous across element faces. *`G_ij`* — the six symmetric geometric factors Neko stores per quadrature point, each the metric tensor contracted with the Jacobian determinant **and the quadrature weight `w3(i,j,k)`** (`src/sem/coef.f90:1130`). *`lx`* — points per element per direction, so polynomial order `lx-1`.

---

## Summary of findings

**The pressure operator (Part 1):**

- On a channel with periodic streamwise/spanwise directions and no-slip walls, the operator Neko's Krylov solver applies is **symmetric to machine precision and positive semi-definite**, with a one-dimensional null space of constants. `[M]`
- The belief that "the SEM pressure operator is not SPD" is **correct for any configuration with a strong pressure boundary condition**, because Neko masks the operator output only — it applies `P·A`, not `P·A·P`, where `P` zeroes the constrained dofs. The asymmetry is 2.3e-2 relative, far above round-off. `[M]`
- On a **tensor-product** mesh the operator is **exactly** the separable Kronecker sum of one-dimensional SEM stiffness and mass operators — to 4e-14. Separability requires a tensor product, **not** uniform spacing: the test mesh is non-uniform in two of three directions. `[M]`
- A direct separable (fast-diagonalisation) solve reproduces Neko's GMRES answer to its tolerance, and is exact: residual 8e-14 against GMRES's 1.3e-8 after 33 preconditioned iterations. `[M]`
- The velocity Helmholtz operator is separable too, and stays so under the no-slip wall mask. `[M]`

**A specialised channel code (Part 2):**

- Neko's operator kernels are already heavily optimised — runtime autotuning across five formulations, fp64 tensor cores, Hopper TMA staging — and are **bandwidth-bound**. The opportunity is in their *inputs*, not their arithmetic. `[C]`
- On an axis-aligned tensor-product box mesh, three of the six `G_ij` are identically zero and the other three reduce to **one scalar per element**. `[C]` If the mesh is additionally uniform in the two homogeneous directions — as a channel mesh is, though the Part 1 test mesh deliberately is not — those collapse further to **one set per wall-normal element layer**. `[C]`
- **Neko already contains a switched-off prototype of exactly this compression.** `[C]` Enabling it is the cheapest way to calibrate everything else here.
- A realistic band is **~2x** on time-to-solution for a structured-metric rewrite keeping the same algorithms, and **~2.5-3.8x** more elements per GPU depending on whether statistics are collected. `[E]` A direct separable pressure solve could go further but trades nearest-neighbour halo exchange for global transposes.
- Several parts of the argument weaken or fail for an LES: a spatially varying eddy viscosity forces a different, non-separable velocity operator. `[C]`

---

## Part 1 — The Pn-Pn pressure operator

### 1.1 What the operator actually is

Neko's `pnpn` scheme is the equal-order (Pn-Pn) velocity–pressure splitting. Its pressure residual routine sets the operator coefficients `[C]`:

```fortran
! src/fluid/bcknd/cpu/pnpn_res_cpu.f90:74-78
c_Xh%h1(i,1,1,1) = 1.0_rp / rho_val
c_Xh%h2(i,1,1,1) = 0.0_rp
c_Xh%ifh2 = .false.
```

`h1` depends on a single scalar — not on the velocity, not on `dt`, not on `mu`. **The pressure operator is `(1/ρ)·L` with `L` the SEM Laplacian and a static geometry.** Everything complicated about Pn-Pn (the curl-curl term, the surface/Neumann terms) lives in the **right-hand side**, not the operator. This is specific to Pn-Pn: the Pn-Pn-2 (Uzawa) splitting instead inverts `D B⁻¹ Dᵀ` on a staggered pressure space, which is a different and not obviously separable operator. Nothing here transfers to it.

Inside the Krylov loop the operator is applied as this triple `[C]` — identically in GMRES (`src/krylov/bcknd/cpu/gmres.f90:246-248`), CG (`src/krylov/bcknd/cpu/cg.f90:207-209`) and the projection basis (`src/common/projection.f90:351-353`):

```fortran
call Ax%compute(w, z, coef, msh, Xh)   ! element-local Helmholtz
call gs_h%op(w, n, GS_OP_ADD)          ! direct stiffness summation
call bc_projector%apply(w, n)          ! zero at strong boundaries
```

Write `P` for that last mask (`P` = projector; `M` is reserved below for the mass matrix). The assembled object is `P·A`.

### 1.2 Method

`contrib/prs_operator_test/assemble.f90` runs a real `pnpn` case for several timesteps, re-executes `prs_res%compute` (the routine that defines the operator), then applies the triple above to unit vectors, column by column.

Three choices make the result meaningful rather than circular:

**Dof identity comes from the operator itself.** The unique-dof numbering is obtained by setting `v(i) = i` and running a `GS_OP_MIN` gather-scatter, so it is the same equivalence relation the assembly step uses. It cannot merge dofs the operator treats as distinct, nor split identical ones.

**The inner product is the solver's own.** Since `coef%mult = 1/multiplicity`, summing over unique dofs with one representative copy each is *exactly* the `glsc3(·,·,coef%mult)` product the Krylov solver minimises in. "The matrix is symmetric" therefore means "the operator is self-adjoint in the norm the solver actually uses" — which is the statement that matters.

**Assumptions are measured, not asserted.** Picking one representative copy per unique dof is only valid if the masked output is continuous across elements. `bc_projector%apply` is a plain local index list that propagates nothing, so a boundary zone masking a dof in one element but not its neighbour would break this. The harness measures the discrepancy directly; it comes out at exactly 0 for every case below. It also reports the right-hand-side consistency `sum_unique(b)`, 3e-15. `[M]`

Matrix-free checks (`⟨u,Av⟩` vs `⟨v,Au⟩`, `⟨v,Av⟩`, `A·1`) run at sizes too large to assemble densely.

**Case A — the channel.** `2π × 2 × π`, periodic in x and z, no-slip walls in y. Deliberately **non-uniform in x** (element widths 1.65 / 2.65 / 1.98) and tanh-stretched in y (height ratio 8.1). `lx = 5`, 54 elements, 3600 unique dofs. Operator state read from the live objects: `h1 ≡ 1.0`, `h2 ≡ 0`, `ifh2 = F`, `prs_dirichlet = F`, pressure mask empty.

**Case C — the same, larger and higher order.** 4 × 8 × 3 elements, non-uniform in x (ratio 2.6), y ratio 14.6, `lx = 8`. 33,516 unique dofs; matrix-free checks only.

### 1.3 Result: symmetric, positive semi-definite `[M]`

Case A:

| quantity | value |
| --- | --- |
| `max abs(A − Aᵀ) / max abs(A)` | **1.25e-16** (machine eps = 2.22e-16) |
| `norm_F(A − Aᵀ) / norm_F(A)` | 6.44e-17 |
| matrix-free `⟨u,Av⟩` vs `⟨v,Au⟩`, relative | 1.57e-16 |
| `abs(A·1)_inf` | 2.84e-14 |
| negative eigenvalues | **0** |
| zero eigenvalues | **1** (the constant) |
| 2nd smallest / largest eigenvalue | 9.33e-03 / 1.07e+02 |
| `max abs(Im λ) / max abs(Re λ)`, full nonsymmetric eigensolve | 1.17e-17 |
| condition number over the non-null modes | 1.14e+04 |

Case C, matrix-free: relative asymmetry 3.45e-16, `⟨v,Av⟩ = 2.3e+04 > 0` on every sample, `abs(A·1)_inf = 1.24e-14`, representative-copy discrepancy 0. *(The sample vectors are randomly seeded, so the asymmetry figure varies between runs at the 1e-16 level.)*

**The operator is SPD on the complement of the constants.** CG is admissible; the constant mode is what Neko's `ortho` removes.

### 1.4 Result: with a Dirichlet pressure condition it genuinely is not symmetric `[M]`

**Case B** — a different mesh: 3 × 4 × 3 elements, uniform in x and z, tanh-stretched over 4 wall-normal layers, **x non-periodic** with a prescribed-velocity inflow on one face and an `outflow` (Dirichlet pressure) on the other. `lx = 5`, 2652 unique dofs.

| quantity | value |
| --- | --- |
| total unique dofs | 2652 |
| `max abs(A − Aᵀ) / max abs(A)` | **2.26e-02** — not round-off |
| all-zero rows (masked) / all-zero columns | 204 / 0 |
| unmasked dofs | 2448 |
| restricted to the 2448 unmasked dofs: symmetry | 1.54e-16 |
| restricted: min / max eigenvalue | 9.24e-04 / 3.21e+01 |
| restricted: negative / zero eigenvalues | 0 / 0 |
| `P·A·P` symmetry | 1.54e-16 |

`scalar_bc_projector%apply` zeroes the operator **output** only, so Neko hands the Krylov solver `P·A`, which is genuinely non-symmetric; the asymmetry sits entirely in the masked columns. Restricted to the unmasked dofs the operator is symmetric and **strictly** positive definite — the Dirichlet condition removes the constant null mode.

Two qualifications, in opposite directions:

- The iteration plausibly never sees the asymmetry. The initial residual is masked (`fluid_pnpn.f90:857`) and every operator output is masked (`gmres.f90:248`), so every Krylov vector is masked. Note the invariant object is `P·A`, not the subspace `range(P)` — `A` alone maps a masked vector out of it, which is precisely why the mask must be reapplied after every operator call. `[C]`
- That argument requires the preconditioner to preserve the mask too, and the CPU HSMG V-cycle does **not** mask its output on exit (`pc_hsmg.f90:621-623`). `[C]` **This was not tested here.** It would need `max abs(z(mask))` measured over the real iterates.

So: the operator underlying a Dirichlet-constrained pressure problem is SPD, but the matrix Neko applies is not symmetric, and whether that matters depends on the preconditioner. GMRES is the safe default for reasons beyond this too — Neko's multigrid preconditioners are not symmetric operators in general.

### 1.5 Result: the operator is exactly separable `[M]`

Index the global dofs `(ix, iy, iz)` with `ix` fastest (matching Neko's `(i,j,k,e)` storage). Build the one-dimensional assembled SEM stiffness `K` and diagonal GLL mass `M` per direction from first principles and compare against

> `A ≟ Mz ⊗ My ⊗ Kx  +  Mz ⊗ Ky ⊗ Mx  +  Kz ⊗ My ⊗ Mx`

| quantity | value |
| --- | --- |
| `max abs(A − A_sep) / max abs(A)` | **4.26e-14** |
| `norm_F(A − A_sep) / norm_F(A)` | 3.20e-14 |

**Separability requires a tensor-product mesh, not a uniform one.** The test mesh is non-uniform in two of three directions and the identity still holds to round-off. Uniformity matters only for the *transform* (§1.8), not the *structure*.

The mechanism: because the quadrature weight is already folded into `G_ij` (`src/sem/coef.f90:1130`), an axis-aligned box element has `G12 = G13 = G23 ≡ 0` exactly and `G11 = s_e · w3(i,j,k)` with `s_e` a single scalar per element. `[C]`

### 1.6 Result: a direct separable solve reproduces Neko's answer `[M]`

Fast diagonalisation (Lynch–Rice–Thomas): solve `K v = λ M v` per direction; the three-dimensional eigenvalues are `λx + λy + λz`; exactly one is zero (0+0+0, the constant) and is dropped.

A smooth manufactured pressure field `p*`, right-hand side `b = A p*` formed through Neko's own operator, solved two ways:

| | residual `norm(Ax − b)/norm(b)` | error vs exact |
| --- | --- | --- |
| Neko GMRES + HSMG, 33 iterations | 1.31e-08 | 3.71e-09 |
| **direct fast diagonalisation** | **8.09e-14** | **8.05e-14** |
| agreement between the two | | 3.71e-09 |

A gauge detail worth knowing if you implement this: dropping the zero fast-diagonalisation mode fixes the constant by the **mass-weighted** condition `⟨1, M x⟩ = 0`, whereas Neko's `ortho` subtracts an unweighted mean over the redundant point count (`src/math/operators.f90:350-368`). `[C]` The two differ by a constant, which is irrelevant to the pressure gradient but will make a naive field-by-field comparison look wrong.

**The operator is one fixed matrix for the whole run.** Assembled after 4 steps and after 9 steps it is bit-identical. `[M]` That is not independent evidence — it follows necessarily from `h1 = 1/ρ` with constant ρ and a static mesh — but it is the property that makes a one-off setup-time diagonalisation viable. It fails under ALE, where `fluid_pnpn.f90:797-812` recomputes the metrics each step. `[C]` *(It does **not** fail under the stress formulation: that changes the velocity operator, while the pressure residual still sets `h1 = 1/ρ`.)*

### 1.7 Result: the velocity Helmholtz operator is separable too `[M]`

At the velocity solve `h1 = μ` and `h2 = ρ·bd/dt`, both spatially constant (they are filled from scalars, `pnpn_res_cpu.f90:212-213`) `[C]`. They change with `dt`, so the velocity operator is constant in time only at fixed timestep.

Assembling the scalar Helmholtz that `Ax_vel%compute_vector` applies per component, with the real per-component no-slip mask (288 masked rows = exactly the two wall planes), **and restricting to the 3312 unmasked dofs**:

| quantity | value |
| --- | --- |
| `max abs(A − Aᵀ) / max abs(A)` | 1.19e-19 |
| min / max eigenvalue | 6.35e-02 / 1.46e+01 (condition number 230) |
| negative eigenvalues | 0 |
| `max abs(A − (μ·Lap_sep + h2·Mass_sep)) / max abs(A)` | **2.85e-15** |

*(The asymmetry is below unit roundoff because the mass term `h2·M` dominates and is exactly diagonal; the symmetric stiffness part carries the eps-level error. Compare §1.3's 1.25e-16 for the pure Laplacian.)*

Removing the two wall planes is a one-dimensional restriction in y, so the tensor structure survives the Dirichlet condition. Both solves in the timestep are therefore directly invertible on a channel mesh — subject to §1.9. This is the per-component operator; the full velocity solve additionally applies `rotate_cyc` (a no-op without cyclic boundary conditions) and solves the three components together.

### 1.8 Result: an FFT is available only where the element spacing is uniform `[M]`

GLL points are non-uniform *inside* an element, so a plain FFT over all points is not available. But on a **uniform periodic element line** the assembled one-dimensional operator is **block-circulant** with block size `lx−1`, so a DFT *across elements* block-diagonalises it:

| | off-block-diagonal magnitude after DFT across elements, relative |
| --- | --- |
| uniform spacing | **1.23e-16** |
| non-uniform spacing (control) | 1.59e-01 |

The 1-D mass diagonal is identical across blocks (spread exactly 0), so the generalised eigenproblem decouples the same way. A periodic direction with uniform elements therefore costs an FFT of length `n_elem` plus a small dense `(lx−1)` solve per wavenumber, rather than a dense `N×N` transform.

Channel meshes are uniform in the two homogeneous directions, so this applies to them — the Part 1 test mesh is deliberately non-uniform in x precisely to separate this condition from the separability condition in §1.5. The wall-normal direction has no such structure and stays a dense transform of size `n_y`; for a `Re_τ = 550` channel `n_y` is a few hundred, so that transform, not the FFTs, sets the cost.

### 1.9 What breaks these results

| condition | pressure operator | velocity operator |
| --- | --- | --- |
| non-tensor-product / curved mesh | separability lost (`G12,G13,G23 ≠ 0`) | lost |
| variable ρ | lost | lost |
| LES eddy viscosity (`nut_field`) | **survives** (operator uses only `1/ρ`) | **lost** — see below |
| SVV | survives | lost (`h1` becomes a full field, `spectral_vanishing_viscosity.f90:151`) |
| ALE / moving mesh | lost (metrics recomputed each step) | lost |
| Dirichlet pressure bc | SPD on the unmasked subspace; `P·A` not symmetric | n/a |

The LES row is the sharpest practical limit `[C]`: setting `case.fluid.nut_field` without `full_stress_formulation` is a hard error (`fluid_pnpn.f90:311-316`), and the stress formulation switches `Ax_vel` to `ax_helm_full`, a coupled operator on the nine Jacobian metrics rather than the six `G_ij`. **A constant-`h1` velocity solve describes a DNS, not a wall-modelled LES.**

### 1.10 Reproducing

`contrib/prs_operator_test/` in this fork. `./mkmesh.sh`, then `./assemble channel.case` (case A), `./assemble channel_order7.case` (case C) or `./assemble outflow.case` (case B), then the Python analysis scripts. Serial, CPU, double precision only; the program refuses to run otherwise. The README carries the full result tables.

---

## Part 2 — A specialised channel code: what could be optimised

**Everything in this part is static analysis and modelling, not measurement.**

**Target configuration.** Turbulent channel, `pnpn`, `lx = 8` (order 7), dealiasing on (`lxd = 12`), fp64, no solution projection, p-multigrid pressure preconditioner, Jacobi + coupled CG velocity. Note this is *not* the configuration of Neko's own `examples/turb_channel`, which uses order 5 (`lx = 6`), HSMG, GMRES(30) and a projection space of 5. Where the reference case is cited below for element counts, the difference is flagged. Order 7 is chosen because Neko's own guidance is that accelerators want at least seventh-order polynomials (`doc/pages/user-guide/performance.md`), and the geometry argument strengthens with `lx`.

### 2.1 The starting point: the operators are already good, and already bandwidth-bound

Neko's own documentation quantifies this `[C]`:

> "at `lx = 8` an element is nine cubes of 4 kB, of which the seven geometric factors alone are 78% of the read traffic" — `doc/pages/user-guide/performance.md:188`
>
> "`Ax` is strongly bandwidth bound, roughly 1.6 flop/byte at `lx = 8` against a ridge point near 9 on GH200" — `performance.md:298`

Two clarifications. The "seven geometric factors" are the six `G_ij` plus `h1`, which is stored as a full field even when constant. And the output cube `w` is write-only, so only eight of the nine cubes are read: the seven factors are 7/8 = 87.5% of **read** traffic, and 78% is 7/9, their share of **total** traffic. The doc's own staging table (`performance.md:213-220`) lists the scalar `Ax` as 8 batched copies, consistent with eight reads.

The kernels are already autotuned at runtime across five formulations — 1-D, kstep, fp64 tensor-core (DMMA), Hopper TMA-staged, and matrix-core (MFMA) on HIP — with the winner cached per operator and polynomial order (`src/math/bcknd/device/cuda/ax_helm.cu:87-231`; the HIP MFMA variant is in `src/math/bcknd/device/hip/ax_helm.hip`). `[C]` **Do not expect to beat these kernels by hand.** Their *inputs* are the opportunity.

### 2.2 The structured-grid collapse

For an axis-aligned tensor-product box mesh (§1.5):

- `G12 = G13 = G23 ≡ 0`
- `G11 = w3(i,j,k) · s_e` — one scalar per element; and if the two homogeneous directions are uniformly spaced, every element in a wall-normal layer is congruent, so **one scalar per y-layer**
- `jacinv`, `B`, `Binv`, `h1`, `h2` likewise; `mult` is separable into a length-`lx` array; the dof coordinates become three 1-D arrays

So `Ax` reads one cube (`u`) and writes one, instead of reading eight and writing one: **nine cubes down to two, a 4.5x traffic cut**, lifting arithmetic intensity from 1.6 to ~7.2 flop/byte. That is still below the ~9 ridge, so the roofline does not cap it — the bandwidth-only ceiling would be 9/1.6 = 5.6x, and the model's own prediction is the full ~4.5x. `[E]`

**Discount to ~3-4x in practice** for kernel-launch overhead, imperfect L2 reuse of the compressed factors, and the non-`Ax` work in the same kernels. That discount is a judgement, not a roofline bound, and it is unmeasured.

**Neko already contains a prototype of this.** `coef_generate_geo_compressed` (`src/sem/coef.f90:1402`) deduplicates identical `G` tensors and builds a `compression_inds` lookup. The call is commented out at `coef.f90:445` and no kernel reads `Gij_compressed`. `[C]` For the reference channel (`examples/turb_channel`, 5832 elements = 18³, so 18 wall-normal layers) it would compress 5832 → 18. Its matcher uses an absolute tolerance of `1e-7` and is quadratic in element count in the worst case, which is plausibly why it was parked. A related path is scoped but unimplemented: `metric_sp_safe` / `NEKO_METRIC_COND_SP` (`coef.f90:71,119`) gate "a future reduced precision storage path". `[C]`

> **The single recommendation in this document.** Before forking anything, enable the existing compression in upstream Neko, replace its matcher with a wall-normal-layer key, and add one `compression_inds[e]` indirection to the element base pointer in the existing kernels. The compressed factor set is then L2-resident (18 layers × 8³ points × 6 arrays × 8 B ≈ 440 kB at `lx = 8`, against 40-50 MB of L2), the autotuner and all backends keep working, and the measurement calibrates every estimate below. Perhaps two weeks of work.

### 2.3 The memory ledger `[C]`

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
| gather-scatter | 2.1 | index arrays, 16 B per gs dof |
| **total** | **245.3** | **1962 B per GLL point** |

At 70 GB usable (an 80 GB device with ~12% headroom for allocator slack, halo buffers and MPI) that is **~69,700 elements**. With BiCGStab instead of GMRES(30): 190.3 FE → ~89,800 elements (**+29%**). GMRES(30) is a case-file choice, not a code limitation — `fused_cg`, `fused_coupled_cg`, `pipecg`, `cheby` and `bicgstab` all exist. `[C]`

**A production run holds more than this.** `fluid_stats` with `set_of_stats: "full"` keeps **58 resident fields** (44 statistics + 9 gradients + 5 work, `src/fluid/fluid_stats.f90:192,201-227`) `[C]`. That is comparable to the GMRES workspace and larger than every other block, it is the scientific output the run exists to produce, and **no optimisation here removes it.**

Note what dominates: the two largest reducible blocks are the dealiasing workspace and the Krylov workspace, **not** the geometry. `coef_t` is 34/245 ≈ 14% of the ledger, or 27% counting the dealiasing metrics.

### 2.4 Opportunities, ranked

**1. Collapse the geometric factors.** §2.2. Affects `Ax`, `opgrad`, `dudxyz`, `cdtp`, `conv1`. ~3-4x on those kernels `[E]`.

**2. Fuse the dealiased convection.** `adv_dealias.f90` carries a source comment calling its device path *"extremely primitive and unoptimized"* (`:253`) `[C]`. Per timestep it does three GLL→GL interpolations, then per velocity component an `opgrad` at the dealiased order, a `vdot3`, a GL→GLL map and a `sub2` — with 17 arrays of persistent GL-order storage. A fused kernel removes nearly all of that traffic and 57 FE of memory. Neko already has the cheaper convecting-field-in-`rst` formulation (`set_convect_rst`, `conv1`) but wires it only into the OIFS path. `[C]`

**3. Improve the curl-curl in the pressure residual — but do not expect to fuse it.** Each `curl` is six separate `dudxyz` calls plus `sub3`s, which can become one kernel per curl. But `curl` **ends with a full gather-scatter** — `opcolv(B) → gs_h%op(ADD) → opcolv(Binv)` (`src/math/bcknd/cpu/opr_cpu.f90:173-178`) `[C]` — so the two successive curls cannot be merged into a single kernel.

**4. Direct-address the gather-scatter.** On a lexicographic structured grid the index arrays become computed offsets. Sizing: at `lx = 8` a hex element has `8³ − 6³ = 296` of 512 dofs on its surface, and the gs metadata is 16 B per gs dof, so DSS moves a data volume of the same order as `Ax` itself. But Neko already detects contiguous runs (`gs_find_blks`) and already fuses the three velocity components into one halo round (`gs_op_vector3`) `[C]`, so this is "remove the residual index loads", not "replace the indirection".

**5. Drop the Krylov workspace.** Configuration, not engineering: BiCGStab (7 FE) or CG (4 FE) instead of GMRES(30) (62 FE). Since the operator is SPD (Part 1), CG is admissible on a channel provided the preconditioner is symmetric.

**6. Mixed precision.** The `rp` kind is fixed at build time (`src/config/num_types.f90.in`), and the default `--enable-real=dp` build sets the extended-accumulation kind `xp` equal to `rp`, so it is uniformly fp64. `[C]` There is **no** path for an fp32 preconditioner inside an fp64 solve — the standard nekRS technique, where the multigrid V-cycle runs in single precision inside a double-precision Krylov iteration. If the V-cycle is bandwidth-bound, fp32 roughly halves its traffic; overall that is ~1.3x if the V-cycle is half the timestep, ~1.5x if it is three-quarters. `[E]`

**7. Reduce host synchronisation in the Krylov loop.** `fusedcg_device_solve` performs **three** host-returning global reductions per iteration (two `device_glsc3` plus `device_fusedcg_part2`, each with its own `MPI_Allreduce` on a non-device-MPI build). `[C]`

### 2.5 The coarse-grid solve

`tamg_device_matvec_flat_impl` (`src/multigrid/tree_amg.f90:586`) implements a *coarse*-level matvec by mapping the coarse vector back to the **finest level of the AMG hierarchy**, applying the full `Ax` there plus two gather-scatters and two masked gather/reduction passes, then mapping back. `[C]` The AMG hierarchy reduces vector length but **not work per matvec**; there is no assembled coarse matrix anywhere in `tree_amg`.

The cost is latency, not bandwidth. Where tree-AMG is used it sits below p-multigrid's coarsest level, `lx = 2`, so each of those `Ax` applications touches `2³ = 8` points per element — about 1.6% of a fine-grid `Ax`. With the defaults (3 AMG levels, Chebyshev degree 4) one coarse solve issues on the order of 20 such matvecs, each several kernel launches: ~150 launches of near-empty kernels per preconditioner application. `[E]`

Part 1 makes this removable — but state the replacement precisely. tree-AMG's own coarse operator is a Galerkin operator on agglomerated aggregates and is *not* separable. What is separable is the **`lx = 2` SEM operator** that tree-AMG is being used to solve. A direct separable solve replaces the whole tree-AMG stack at that level, rather than accelerating it.

**How much this matters depends on the preconditioner**, and the two options differ sharply:

- **p-multigrid (`phmg`)** — the V-cycle is `Ax`-based: at the default `smoother_iterations = 3`, each level costs 2 `Ax` on the down leg (the first Chebyshev iteration runs with a zero initial guess and needs no residual), 1 for the residual, and 3 on the up leg — **6 per level per cycle** `[C]`. Both the geometry collapse and the coarse-solve replacement pay directly here.
- **HSMG** (Schwarz + FDM, Neko's default for its own channel example) — the V-cycle contains **no `Ax` application** outside the coarse solve; it is roughly ten halo exchanges, two overlapping Schwarz solves and four interpolations `[C]`. Collapsing `G_ij` speeds up only the single `Ax` per Krylov iteration. Its coarse solver also defaults to CG + Jacobi with 10 iterations on the `lx = 2` grid, not tree-AMG (`pc_hsmg.f90:161-168`) `[C]`, so the latency argument above applies to `phmg` and to explicitly-configured tree-AMG, not to stock HSMG.

Two further HSMG-specific points `[C]`: it carries per-element geometry the ledger does not count — FDM's `s` and `d(nl³, nelv)` with `nl = lx+2` (`src/math/fdm.f90:124-126`), roughly 5 FE at `lx = 6` per Schwarz level, which *also* compresses by layer on a structured mesh and is therefore additional upside for that configuration. And `pc_hsmg.f90:267` calls `msh%all_deformed()`, globally marking every element deformed and disabling the only axis-aligned fast path in the code — which exists on the CPU backends only (`ax_helm_xsmm.F90`, `pc_jacobi.f90`), never in CUDA.

### 2.6 A direct separable pressure solve: the large prize, and its price

Part 1 establishes the operator is exactly separable and constant in time, so a fast-diagonalisation solve needs **no Krylov iteration at all**.

Note that *exactness* is not itself the prize — the reference case asks only for `absolute_tolerance: 1e-3` on the pressure, and the splitting error bounds accuracy anyway. The prize is that the iteration count goes to zero and with it the whole preconditioner stack.

Cost, with FFTs across elements in the two uniform periodic directions (§1.8) and a dense transform of size `n_y` in the wall-normal direction: on the order of a few `Ax`-equivalents of dense GEMM per solve, against O(100) `Ax`-equivalents for a preconditioned Krylov solve. `[E]`

**The price is the communication pattern, and it is the single biggest risk to any speedup estimate.** A direct solve needs a global transform in each direction, i.e. pencil transposes (all-to-all) per solve, replacing Neko's nearest-neighbour halo exchange. That scales like a classical spectral channel code, not like Neko. On one to a few GPUs it is free; beyond that it is the dominant question and **has not been measured here**.

Two practical obstacles `[C]`: the operator is singular (pure Neumann; null space removed by `ortho`), and constant-mass-flux forcing runs a second full pressure solve — plus a full coupled velocity solve — whenever the base flow is recomputed, which happens on any change of `dt` or of the BDF coefficient (`src/fluid/fluid_volflow.f90:385-398`, solves at `:242` and `:298`). That is once on a fixed-`dt` run but frequently under variable timestepping, and those solves would have to go through the direct path too.

### 2.7 Costs a naive model ignores

A real `pnpn` timestep contains substantial work that is neither `Ax` nor gather-scatter `[C]`:

- **CFL is computed twice per step** with variable timestepping (`simulation.f90:135,137`), each a 13-array full-field pass plus an `MPI_Allreduce`.
- **Two global reductions per step** for the pure-Neumann pressure (`ortho` at `fluid_pnpn.f90:848,891`), plus **two unconditional global barriers** in the output controller even when nothing is written (`output_controller.f90:247,296`).
- **Five gather-scatter rounds in residual assembly alone**, before any Krylov iteration.
- **Simulation components run every step by default** — in the reference channel case `lambda2` (a full 9-component gradient tensor plus a 3×3 eigenvalue solve per point) has no `compute_control` and so runs every step.
- Constant-mass-flux forcing on recompute (§2.6), source terms, weak boundary conditions, EXT/BDF assembly, lag rotation, material properties.

### 2.8 Estimates, with error bars

| | estimate | basis |
| --- | --- | --- |
| `Ax`-class kernels, structured metrics | ~4.5x modelled, **~3-4x** assumed | `[E]` bandwidth model, then a judgement discount — not measured |
| dealiased convection, fused | 5-10x on that term | `[E]` low confidence |
| **time-to-solution, structured rewrite, same algorithms** | **~2x** | `[E]` **unmeasured; treat as a hypothesis** |
| plus fp32 preconditioner | ~1.3-1.5x further | `[E]` conditional on the V-cycle's share |
| plus direct separable pressure solve, 1-4 GPUs | substantially more; not bounded here | `[E]` transposes unmeasured |
| elements/GPU, no statistics collected | **~3.8x** | `[C]` arithmetic, shown below |
| **elements/GPU, with `fluid_stats` "full" resident** | **~2.5x** | `[C]` arithmetic, shown below |

The element-count arithmetic, from §2.3's ledger. A specialised code removes `coef_t` (34.0 → ~0), the device dofmap (3.0 → ~0), the dealiasing block (57.4 → ~0), the gather-scatter metadata (2.1 → ~0) and the per-level multigrid geometry (9.8 → 8.4); swaps GMRES(30) for BiCGStab (62.0 → 7.0); and drops the 10-deep fused-CG search space for a plain coupled CG (41.0 → ~13). The solution state (36.0) is irreducible. Total **64.4 FE against 245.3 — a factor 3.8**. Adding `fluid_stats` at 58 FE to both sides: 122.4 against 303.3, **a factor 2.5**.

**The memory numbers are arithmetic and are reliable. The time-to-solution numbers are not measured and should be treated as a hypothesis to test.** The way to calibrate them is the recommendation in §2.2.

### 2.9 Effort `[E]`

All figures below are estimates for one experienced GPU/CFD developer, and none is measured.

The CUDA layer (~21k lines) lifts cleanly — `ax_helm_kernel.h` needs only `elem_block.h`, the DMMA headers and `device_config.h`. The Fortran layer (~111k lines in the core directories) is where the abstraction lives, and almost none of it is wanted; expect to write 5-8k lines of new driver.

- Correct structured `pnpn` channel solver on one GPU, reusing Neko's kernels: **~2-3 months**
- Structured metrics, fused convection, direct-addressed gather-scatter, fp32 preconditioner, multi-GPU, validated against Neko: **~9-12 months**
- Direct separable pressure solve: **+2-3 months**

---

## Open questions

1. **Does the geometry collapse actually deliver 3-4x on `Ax` in situ?** Testable cheaply inside Neko (§2.2). Everything downstream depends on it.
2. **Is `P·A` mask-invariant under every preconditioner?** §1.4 establishes that the Krylov vectors are masked, but notes `pc_hsmg.f90:621-623` does not mask on exit. Measure `max abs(z(mask))` over real iterates.
3. **What do the pencil transposes cost at the intended machine size?** The decisive unknown for the direct-solve path, and the reason no scaling claim is made here.
4. **Where does the time actually go per timestep on a GPU?** No profile informed this document. §2.7 lists costs a bandwidth model ignores; some may dominate.
5. **Is any of this compatible with the intended physics?** If the target is a wall-modelled LES rather than a DNS, the velocity operator is not separable and `h1` is not constant (§1.9).

## Key source references

| topic | location |
| --- | --- |
| pressure operator definition | `src/fluid/bcknd/cpu/pnpn_res_cpu.f90:74-78` |
| operator application in the Krylov loop | `src/krylov/bcknd/cpu/gmres.f90:246-248` |
| geometric factors, weight folded in | `src/sem/coef.f90:1130` |
| geometric-factor compression (disabled) | `src/sem/coef.f90:445, 1402` |
| roofline and autotuner documentation | `doc/pages/user-guide/performance.md:113-300` |
| dealiased advection | `src/fluid/bcknd/advection/adv_dealias.f90:107-253` |
| gather-scatter inside `curl` | `src/math/bcknd/cpu/opr_cpu.f90:173-178` |
| matrix-free AMG coarse matvec | `src/multigrid/tree_amg.f90:586` |
| p-multigrid V-cycle and defaults | `src/multigrid/phmg.f90:134-186, 616` |
| HSMG coarse solver defaults | `src/krylov/pc_hsmg.f90:161-168` |
| GMRES workspace | `src/krylov/bcknd/device/gmres_device.F90:65, 174-175` |
| gather-scatter indirection | `src/gs/bcknd/device/cuda/gs_kernels.h:43-66` |
| precision kinds | `src/config/num_types.f90.in` |
| LES forces the stress formulation | `src/fluid/fluid_pnpn.f90:311-316` |
| statistics memory | `src/fluid/fluid_stats.f90:192, 201-227` |
| test harness | `contrib/prs_operator_test/` |
