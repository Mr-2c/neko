# Neko: verified improvement opportunities

A curated set of defects, performance opportunities and configuration gaps found while reading [Neko](https://github.com/ExtremeFLOW/neko) closely. Everything here applies to **Neko as it stands, on general unstructured meshes** — nothing depends on a structured or box mesh.

**Provenance.** Neko 1.99.9, based on upstream commit `ce9b260` (September 2026). Every item was verified against the source and then independently re-checked by a second reader; line numbers are from that tree. Items that did not survive the second pass were dropped, and the most interesting of those are listed in *Checked and not a problem* at the end — that section exists because knowing what was already ruled out is worth as much as the findings.

**How to read an entry.** Each carries a severity, an effort estimate, and — deliberately — **the strongest objection a maintainer could raise**. Several of these are defensible design decisions rather than oversights, and the entry says so where that is true. Nothing here was measured on a GPU; performance figures are traffic/launch counts and models, labelled as such.

---

## At a glance

| # | Item | Kind | Effort |
| --- | --- | --- | ---: |
| A1 | Controller assignment drops `never` — a "never" controller fires every step | **defect** | trivial |
| A2 | `set_counter` divides by zero for a controller with no time interval | **defect** | trivial |
| A3 | `relative_tolerance` is in the schema, parsed by nothing, used by nothing | **defect** | small |
| A4 | `jacobian_inverse` inverts a hardcoded point index | defect (dead code) | trivial |
| B1 | Gather-scatter facet loop runs with half the warp masked off | perf | trivial |
| B2 | Local gather and scatter are two kernels over the same indices; ×6 for a vector gs | perf | medium |
| B3 | Dealiased advection re-reads all 9 GL metrics per velocity component | perf | medium |
| B4 | The double curl runs 18 tensor contractions where 9 suffice | perf | medium |
| B5 | tree-AMG coarse matvec applies the full finest-level operator at every level | perf | large |
| B6 | Three blocking host-returning reductions per device CG iteration | perf | medium |
| B7 | No undeformed-element fast path in any device `Ax` kernel | perf | large |
| C1 | GMRES restart length hardcoded at 30, unreachable from the case file | config | small |
| C2 | `lambda2` computes every step; three shipped examples leave it there | config | trivial |
| C3 | Two unconditional MPI barriers per step for a discarded timing line | perf | trivial |
| C4 | Pressure projection is suspended after every timestep change | doc | trivial |
| D1 | `coef_generate_geo_compressed`: unreachable, quadratic, with an out-of-bounds write | maint | small |
| D2 | `coef_metric_condition`: un-threaded per-point eigenvalue solve at every init | perf | small |
| D3 | `cpr` exports a public API that cannot do anything | maint | medium |
| E1 | `pc_hsmg` permanently mutates the shared mesh as a construction side effect | design | medium |

---

## A. Defects

### A1. Assigning a `time_based_controller_t` silently drops half its state

`src/common/time_based_controller.f90:199-209` copies five components and omits four:

```fortran
ctrl1%end_time      = ctrl2%end_time
ctrl1%frequency     = ctrl2%frequency
ctrl1%nsteps        = ctrl2%nsteps
ctrl1%time_interval = ctrl2%time_interval
ctrl1%nexecutions   = ctrl2%nexecutions
! start_time, control_mode, control_value and never are NOT copied
```

`never` is the one that bites. `init` handles `control_mode: "never"` by setting **only** `this%never = .true.` (`:125-126`), leaving `nsteps`, `time_interval` and `frequency` at their type defaults of 0. `check()` short-circuits on `this%never`, but the assignment leaves the copy at its default `.false.` (`:61`). Execution then falls through to

```fortran
else if ((this%nsteps .eq. 0) .and. (t .ge. this%nexecutions * this%time_interval - 0.1_dp*dt))
```

which with `time_interval = 0` is true on essentially every step. **A `never` controller, once assigned, fires every step.**

This is live, not theoretical: `src/simulation_components/simulation_component.f90:281-283` assigns all three controllers through this operator, which is the path every simulation component takes.

*Fix:* copy the remaining four components. *Effort:* trivial. *Objection:* none that I can see — the omission looks like an oversight rather than a decision, and `start_time` is read by `check()` too.

### A2. `set_counter` divides by `time_interval` without guarding it

`src/common/time_based_controller.f90:~205`:

```fortran
if (this%nsteps .eq. 0) then
   this%nexecutions = int(((time%t - time%start_time) + 0.1_dp*time%dt) / this%time_interval) + 1
end if
```

`nsteps == 0` and `time_interval == 0` hold together for a `never` controller, and also for any controller that reached this state through A1. The guard tests the wrong component — it should test the interval it is about to divide by, or return early on `never`.

*Fix:* guard on `never` and on a non-zero `time_interval`. *Effort:* trivial. *Objection:* reachable only on restart, and only in combination with A1 — fix A1 first and this becomes defensive.

### A3. `relative_tolerance` is advertised but implemented nowhere

Three independent facts:

- `rel_tol` is a component of `ksp_t` (`src/krylov/krylov.f90:76`) and is plumbed through the `init` of all 19 concrete solvers.
- `krylov_is_converged` (`krylov.f90:410-420`) reads **only** `abs_tol`. No solver's loop exit test reads `rel_tol` either.
- `relative_tolerance` appears in `doc/schemas/common.schema.json:496` and in **no Fortran file at all** — nothing parses it from the case file.

So a user who sets `relative_tolerance` gets a schema-validated case file and no effect whatsoever. A relative criterion is the one most users actually want for a pressure solve, since the residual scale changes with `dt` and with the flow.

The same routine also reports a solve that converged exactly at `max_iter` as not converged (`if (iter .ge. this%max_iter) converged = .false.`), which is an off-by-one in the verdict rather than in the solve.

*Fix:* either implement it (parse the key, and test `residual <= max(abs_tol, rel_tol * res_start)`) or remove it from the schema. Shipping it in the schema unimplemented is the worst of the three. *Effort:* small. *Objection:* implementing it changes iteration counts in existing cases only if users set the key, which today they cannot.

### A4. `jacobian_inverse` inverts a hardcoded point index

`src/sem/local_interpolation.f90:295-313`:

```fortran
do i = 1, n_pts
   tmp = matinv3(real(jacinv(:, :, 3), xp))
   jacinv(:, :, i) = tmp
end do
```

The subscript is the literal `3`. Because the array is overwritten in place, at `i = 3` the slice is replaced by `inv(J₃)`, so every later iteration computes `matinv3(inv(J₃)) = J₃`: points 1-3 receive `inv(J₃)` and points 4..n receive **`J₃` itself, un-inverted**. For `n_pts < 3` the read is out of bounds.

**Runtime impact today is zero** — the routine is private and has no caller. The value of recording it is that it is a trap with a correct-looking name: anyone adding a local Newton solve for `rst` inversion (probes, particles, overset search) will reach for it, get plausibly-shaped 3×3 output at every point, and debug a Newton iteration that converges only near point 3. No test can catch it because it has no caller.

*Fix:* delete it — the live `rst`-inversion path already uses `matinv3x3` at `legendre_rst_finder.f90:338`, so nothing depends on it. If kept, the subscript becomes `i` and it needs a test with **distinct** Jacobians and `n_pts ≥ 4` (a test with `n_pts ≤ 3`, or with identical points, passes with the bug in place). *Effort:* trivial.

---

## B. Performance, general meshes

### B1. The gather-scatter facet loop runs with half the warp masked off

`src/gs/bcknd/device/cuda/gs_kernels.h:72-79`, identical in the HIP kernels and repeated in all four reduction variants:

```c
else {
  if ((idx%2 == 0)) {
    for (int i = ((o - 1) + idx); i < m ; i += str) {
      T tmp = u[gd[i] - 1] + u[gd[i+1] - 1];
      v[dg[i] - 1] = tmp;
    }
  }
}
```

Only even-numbered threads participate, so every warp issues this loop from 16 lanes instead of 32, and the active lanes read `gd[]` with stride 2 — half of every loaded cache line is wasted.

This is not a corner case. The `o > 0` branch is the **local** gather, and `o = local_facet_offset` (`gather_scatter.f90:1212`) means this loop handles the facet dofs — at `lx = 8` roughly 73% of the local gs list, since a hex has 216 face-interior dofs against 72 edge and 8 vertex.

*Fix:* `for (int i = (o-1) + 2*idx; i < m; i += 2*str)` and drop the mask. Index-identical coverage — each thread still owns one consecutive pair — with all 32 lanes active and coalesced reads across the warp. *Effort:* trivial, one line in each of eight kernels. *Objection:* the loads are random gathers, so if the device is already saturating in-flight requests the gain will be well under 2× on that loop; expect single-digit percent of total gs time, more at small per-GPU problem sizes.

### B2. The local gather and scatter are two kernels over the same index arrays

`gather_scatter.f90:1756` and `:1759` issue `gather_kernel_add` then `scatter_kernel` back to back with nothing in between for the local part — no communication, no host work. Both kernels independently read `dg[]`, `gd[]`, `b[]`, `bo[]`, and the gather writes `gs%local_gs` only for the scatter to read it straight back.

The three-component vector gs compounds this: `gs_op_r3` fused the **communication** side properly (`shared_gs_v`, `cuda_gs_pack_vec`/`unpack_vec`) but left the local side as three scalar gathers plus three scalar scatters, so the index arrays are read **six times** per velocity gather-scatter.

*Fix:* a fused `gather_scatter_local` kernel that sums a block and writes it straight back to all its members, with an `nc`-component variant. Index traffic per 3-component gs falls from roughly 28n bytes to 4.6n, and six launches become one. It is structurally the same change the communication side already received. *Effort:* medium. *Objection:* the fused kernel must handle in-place read-modify-write correctly across the block, and the existing separation is what makes the shared/local split easy to reason about.

### B3. Dealiased advection re-reads all nine GL metrics per velocity component

`src/fluid/bcknd/advection/adv_dealias.f90:253` carries the comment `!This is extremely primitive and unoptimized on the device //Karp`. The device branch issues 15 kernels: three GLL→GL interpolations, then per component an `opgrad` at the dealiased order (reads `u` + 9 metrics, writes 3), a `vdot3` (reads 6, writes 1), a map back and a `sub2`. That is 66 accesses the size of a GL field per timestep, **27 of them metric reads** — the same nine arrays read three times.

It also holds 17 persistent GL-order arrays: the 9 metrics of a `coef_GL` (`:118`, via `coef_init_empty`) plus 8 work arrays (`:135-153`). At `lxd = 3(order+1)/2` each is `(3/2)³ = 3.375×` a GLL field.

Neko already contains the cheaper exact reformulation — `set_convect_rst` computes the convecting field once in reference coordinates, and `convect_scalar`/`conv1` then need no metrics at all — but it is wired only into the OIFS path.

This applies to **every incompressible example**: 32 of the 38 shipped case files set `"dealias": true` and none set it false; the six that omit it are the compressible ones, which never construct an `adv_dealias_t`.

*Fix:* call `set_convect_rst` once per timestep and use the `rst` form for all three components. Call the backend routines directly (`opr_device_set_convect_rst` takes bare `c_ptr`s) rather than the public `field_t`-based wrappers, which would allocate internal dofmaps and *increase* memory. *Effort:* medium. *Objection:* summation order changes, so exact-output baselines move; and a maintainer may reasonably say the current code is the deliberately simple, obviously-correct transcription — though the in-tree comment reads as an acknowledged TODO rather than a defence.

### B4. The double curl runs 18 tensor contractions where 9 suffice

`opr_device_curl` (`src/math/bcknd/device/opr_device.F90:822-1011`) issues six separate `cuda_dudxyz` calls plus three `sub3`. Each velocity component appears in two of the six calls, so its three reference derivatives are computed **twice**. Per curl that is 45n of traffic where a fused kernel needs 16n, and 18 contractions where 9 suffice.

The pressure residual runs two curls back to back (`pnpn_res_device.F90:315-316`), and the stress formulation mirrors it. Note the two curls cannot be fused *with each other* — each ends with `opcolv(B) → gs_h%op(ADD) → opcolv(Binv)`, a genuine gather-scatter — but each curl individually can be one kernel.

`lambda2_kernel.h` is a working template: it already loads three components into shared memory and forms all nine derivatives once.

*Fix:* per-backend `curl` kernels on the `lambda2` pattern, folding `opcolv(B)` into the store (3-D branch only). *Effort:* medium. *Objection:* four new per-backend kernels plus a tuner entry, and shared memory grows from one cube to three, lowering occupancy at large `lx`. Measured against a pressure solve of 10-20 preconditioned iterations, residual assembly is not where the time goes — rank this below B3.

### B5. tree-AMG coarse matvecs apply the full finest-level operator

`tamg_device_matvec_flat_impl` (`src/multigrid/tree_amg.f90:586-632`) implements a *coarse*-level matvec by mapping the coarse vector back to the finest level of the AMG hierarchy, applying the full `Ax` there plus two gather-scatters and two masked gather/reduction passes, then mapping back. Coarsening reduces vector length but **not work per matvec**: a level-2 matvec on a vector 64× shorter does the same work as a level-0 matvec. There is no assembled coarse matrix anywhere in the module.

*Scope this carefully.* tree-AMG is reachable only through `phmg`, and across the shipped examples `hsmg` appears 29 times against `phmg` 3 — and `hsmg`'s coarse grid is a CG+Jacobi solve, not tree-AMG (`pc_hsmg.f90:161-168`). So this is the critical path for `phmg` users, not for every pressure solve. Within that scope it is real and mesh-independent, multiplied by the pressure iteration count.

*Fix:* assemble a Galerkin coarse operator per level once at setup and replace the matvec with an SpMV. Two obstacles are real: the operator applied is the composite `D Qᵀ P A P Q D`, not bare `A`; and there is no element-matrix extraction path for `ax_helm` anywhere in the tree — though at `lx = 2` the element block is only 8×8. *Effort:* large. *Objection:* a Galerkin operator from an unsmoothed tentative prolongator is generally worse-conditioned, so iteration counts can rise even as per-iteration cost falls; it must be measured end to end. The assembled matrix also needs a halo scheme the module currently gets free from `gs_h`, and rebuilding on geometry change is the common path, since all three shipped `phmg` cases are moving-mesh.

### B6. Three blocking host-returning reductions per device CG iteration

`cg_device` does `rtz1` (`:210`), `pap` (`:220`) and `rtr` (`:227`) per iteration; `fusedcg_device` has the identical structure. All three block: `device_glsc3` ends in an `MPI_Allreduce` unless built with device MPI, and `cuda_global_reduce_add_xp` ends in `cudaStreamSynchronize` on every path including NCCL and NVSHMEM. `"cg"` is selected by 40 solver blocks across the shipped examples.

Only `rtz1` is genuinely serial — `beta = rtz1/rtz2` feeds the search-direction update before `Ax`. `pap` and the residual norm can be merged into one `glsc3_many` over a pointer triple.

*Fix:* merge `pap` and `rtr` into one multi-vector reduction; optionally defer the convergence test by `k` iterations. *Effort:* medium. *Objection:* the merged residual is the recurrence residual rather than the true one, which is exactly what the third reduction was written to provide; a maintainer would reasonably require a periodic true-residual recompute as a condition. Deferring convergence can overshoot the tolerance by up to `k-1` iterations and changes reported iteration counts in tests.

### B7. No undeformed-element fast path in any device `Ax` kernel

On an axis-aligned element `G12 = G13 = G23 = 0`, so a fast path drops three of the eight read cubes. The CPU side has exactly one such branch — `ax_helm_xsmm.F90:128,149` — and the plain CPU, SX and *all* device backends have none. A grep for `deform|dfrmd` over every `.cu`, `.hip`, `.cl` and `.metal` file returns only the Jacobi preconditioner setup, where the corrections are applied unconditionally (harmless, since the terms are zero and it is setup, not the hot loop).

The ceiling is about 1.5× on the `Ax` kernel and less end to end. It buys nothing on curved meshes.

*The prerequisite is the real finding:* `dfrmd_el` is **not** a trustworthy deformation flag today (see E1), so a fast path needs a new predicate, not the existing one. Derive it from the computed geometry — after `coef_generate_geo`, test per element whether `max(|G12|,|G13|,|G23|)` is below a tolerance scaled by `max|G11|`, and store the result on `coef_t`. That is one cheap setup reduction, immune to curvature, user mesh deformation and ALE, and per-`coef` so multigrid levels get their own answer.

*Effort:* large. *Objection:* the DMMA and TMA variants bake `DMMA_NG = 7` into compile-time shared-memory structs, so covering them means another instantiation on an already large tuner matrix; and a per-element flag on a mixed mesh would want elements sorted by deformation, perturbing an ordering everything downstream depends on. Start with a whole-rank flag.

---

## C. Configuration and defaults

### C1. GMRES restart length is hardcoded at 30

`gmres_device.F90:65` (`m_restart = 30`), `gmres.f90:55` and `gmres_sx.f90:53` (`lgmres = 30`). A grep for any assignment to either name outside the declaration returns nothing — the value is never set anywhere. `krylov_solver_factory` takes no restart argument, and no restart or subspace key exists in the case-file documentation.

The cost is `2·m` full-size work vectors — at `m = 30` that is 60, which in a per-rank memory ledger is the single largest reducible block, larger than the whole geometry object. A user who wants GMRES(10) to fit a bigger mesh, or GMRES(50) for a harder operator, has no way to ask.

*Fix:* the clean route avoids touching the deferred `ksp_init` interface (which would force an identical signature change in all 19 solver extensions and break out-of-tree registrations): read the key in the GMRES constructors and set `m_restart` before the allocations. *Effort:* small. *Objection:* exposing it lets users pick a value that silently degrades convergence, so it wants a documented range. "30 is a tuned default" is an argument for a good default, not for making it unreachable.

### C2. `lambda2` computes every timestep by default

Simulation components default to `compute_control: "tsteps"` with `compute_value: 1` (`simulation_component.f90:351-367`). `examples/turb_channel/turb_channel.case:62` and `examples/recycling/recycling.case:68` declare `lambda2` with no control at all; `TS_channel.case:64` sets `compute_value: 1` explicitly. `lambda2` is visualisation-only — nine contractions plus a 3×3 eigenvalue solve per point, with four to five transcendentals each, and it is one of only two operators with no matrix-unit variant, so it is relatively worse on GPU.

`examples/cylinder/cylinder.case:78-79` already does the right thing: `"compute_control": "fluid_output"`, a first-class mode that redirects to the fluid's own output cadence.

*Fix:* examples and docs only — add `"compute_control": "fluid_output"` to the three cases and to the `simcomps.md` snippet. Do not change the code default. *Effort:* trivial. *Caveat worth stating:* with `output_at_end: true` the final field then carries the `lambda2` from the last fluid-output step rather than the final step. `cylinder.case` already lives with this.

### C3. Two unconditional MPI barriers per step for a discarded timing line

`src/io/output_controller.f90:247-248` and `:296-297` bracket the output section with barriers that fire every step regardless of whether anything is written — and the source itself asks `!Do we need this Barrier?` at `:246`.

Be honest about magnitude: once ranks are tightly coupled by the many Krylov `MPI_Allreduce`s in a step they arrive nearly together, so the marginal cost is close to raw barrier latency — order 10-30 µs at 10⁴ ranks. Against a 10-50 ms step that is noise. It matters only in the extreme strong-scaling regime Neko targets, where step times are 1-3 ms and the pair becomes a few percent — and where it adds two synchronisation points that convert per-rank jitter into shared delay for no reason.

*Fix:* hoist the write decision above the barriers and guard them with it. Safe because `time_based_controller_check` is local arithmetic on rank-replicated state, so the predicate is rank-identical and the guarded collective stays consistent. *Effort:* trivial. *Objection:* a cheap periodic sync point before I/O is defensible, and the benefit is unmeasured.

### C4. Pressure projection is suspended for several steps after every timestep change

Not a defect — a documented interaction worth knowing. With `variable_timestep: true`, `projection_pre_solving` clears the basis on the step `dt` changes and re-engages projection only once `dt_last_change > projection_hold_steps - 1` (`src/common/projection.f90:249-262`, default hold 5). If the timestep adapts more often than the hold length, projection may rarely engage at all, and the user pays for the projection memory (`2L+1` full fields per scalar) without the iteration-count benefit.

*Fix:* documentation. *Effort:* trivial.

---

## D. Dead code

Neko contains a small set of unreferenced procedures. Most are deliberate (`field_add4`/`matrix_add4`/`vector_add4` for API symmetry, the `tamg_print_*` debug helpers in a live module, `tri_mesh_brute_force` as a documented reference implementation) and should be left alone. Three are worth a decision.

### D1. `coef_generate_geo_compressed`

`src/sem/coef.f90:1402-1476` deduplicates identical geometric-factor sets across elements and builds a `compression_inds` lookup — the idea is sound and the payoff on meshes with many congruent elements is large. But the only call site is commented out at `coef.f90:445`, nothing reads the compressed arrays, and as written re-enabling it would be a trap:

- The scan is O(nelv²·lxyz) with no early exit inside the point loop. Transcribing the loop verbatim and timing it gives 0.50 s at 500 elements, 2.07 s at 1000, 8.72 s at 2000 — clean quadratic, so ~35 s at 4k elements and ~14 min at 20k, and **slowest precisely on the general unstructured meshes where it finds nothing**.
- `G11..G23` are never freed and the compressed arrays are never `device_map`'d, so it strictly *adds* memory.
- With `nelv == 0` it writes out of bounds at `:1416` and `:1418`.
- The match criterion is an **absolute** `1e-7` on a sum over `lxyz·9` dimensional terms, so it is unit- and order-dependent: the same mesh can compress at `lx = 4` and not at `lx = 8`, or in millimetres and not in metres.
- It logs with a bare `write(*,*)` from every rank.

*Fix:* delete it, and note in the commit that git history preserves the prototype. If instead it is to be revived, the matcher should sort or hash on the exact bit pattern (a quantised-bucket hash does **not** preserve tolerance semantics — two elements within tolerance can land in adjacent buckets), the tolerance should be relative, and the arrays need freeing and mapping. *Effort:* small to delete; genuinely multi-week to revive and wire, since the indirection threads through the CPU, SX, XSMM, CUDA, HIP and OpenCL `Ax`/`opgrad`/`cdtp` kernels.

*Objection, and it is a fair one:* this is a deliberately parked prototype — the call was commented rather than deleted so the idea is not lost, and `!! @note This could be faster with various tweaks` at `:1401` is the author saying it is unfinished. The narrow counter is the only one worth making: as parked it contains an out-of-bounds write and is slowest where it yields nothing, so re-enabling it as-is is a trap for whoever tries.

### D2. `coef_metric_condition` runs an un-threaded per-point eigenvalue solve at every init

`src/sem/coef.f90:1288-1398` calls `eig_sym3` — one `sqrt`, one `acos`, two `cos` — at every quadrature point, in a four-deep loop with no OpenMP, while the neighbouring `coef_generate_geo` *is* threaded. Timed on the same loop transcribed verbatim: 96.6 ns/point against 4.0 ns/point for a streaming pass, a 24× multiple. A 100k-element rank at `lx = 8` spends ~5 s here, single-threaded on the host, before the first timestep, while the GPU idles.

The degeneracy count it produces is a real, always-active mesh check. But the *eigenvalues* buy only the condition number, which on a double-precision build feeds one log line and a warning gated on `rp == sp` that can never fire.

*Fix:* two independent changes, both cheap. (a) Add OpenMP to the element loop with `reduction(max:)` / `reduction(+:)` — max and sum are order-independent so the result is bit-identical. (b) Split degeneracy detection from conditioning: positive-definiteness needs only Sylvester's criterion on the 3×3 metric, measured at 4.35 ns/point — 22× cheaper and indistinguishable from a pure stream. Keep the existing `scal` normalisation so the triple product does not overflow. *Effort:* small.

*Related:* `metric_sp_safe` and `NEKO_METRIC_COND_SP` are computed and never read. That half **is** intentional — the doc comment says explicitly that it gates "a future reduced precision storage path" — so lead with the un-threaded loop, not with the unused flag.

### D3. `cpr` exports a public API that cannot do anything

`src/sem/cpr.f90` exports `cpr_t`, `cpr_init` and `cpr_free`, and `neko.f90:106` re-exports them. The one operation the module exists for, `cpr_truncate_wn` (`:262-371`), is private and unreferenced, so a user can construct the object, have the field transformed to spectral space, and then has no way to invoke the compression.

Simply making it public would ship an unconditional debug block keyed to a hardcoded element index (`:338-370`, `if (e .eq. 50)`), whose first action logs the string *"Debugging info for e=50. Do not forget to delete"* (`:340`). The error target is also hardcoded and **absolute** (`:284`, `targeterr = 1e-3`), so it is not scale-invariant.

*Fix:* decide. Either deprecate — drop the re-export and say the module is unfinished, rather than leaving a public API that is a no-op — or finish it: remove the debug block, make the target a relative, JSON-driven parameter, export the routine and add a test that truncates a smooth field and asserts the achieved error meets the request. *Effort:* medium. *Objection:* an experimental module parked mid-development is a reasonable thing to keep; the complaint is only that it is *exported*.

---

## E. Design side effects

### E1. `pc_hsmg` permanently flips every element of the shared mesh to "deformed"

`src/krylov/pc_hsmg.f90:266-267`, inside the constructor:

```fortran
! Compute all elements as if they are deformed
call coef%msh%all_deformed()
```

`mesh_all_deformed` sets `dfrmd_el = .true.` for every element of the one shared `mesh_t` (`src/mesh/mesh.f90:498-502`). Nothing restores it — `hsmg_free` does not, and the only routine that ever recomputes the flags runs once at `mesh_finalize`. So **selecting a preconditioner changes the code path taken by the fluid and scalar solves themselves**, for the rest of the run.

Two consequences, and the second is the more interesting:

- *Performance, XSMM builds only.* `ax_helm_xsmm.F90:128,149` is the only `Ax`-path reader of the flag, so on a build configured with libxsmm every Helmholtz apply pays six extra `addcol3` sweeps per element. On plain CPU, SX and all device backends there is **zero** effect — no `Ax` there reads the flag.
- *Reproducibility.* The CPU Jacobi diagonal and the XSMM `Ax` become bitwise dependent on the preconditioner choice, because the added `G12`/`G13`/`G23` terms on a nominally axis-aligned element are round-off-sized rather than exactly zero. Two otherwise identical runs differing only in `preconditioner.type` produce different last digits and different iteration counts.

*Fix:* make the flag trustworthy at its source and then delete the call, rather than relocating it. After `coef_generate_geo`, mark an element deformed when `max(|G12|,|G13|,|G23|)` exceeds a scaled tolerance — exact by construction, catches curvature, immune to user mesh deformation and ALE, and per-`coef` so multigrid levels get their own answer. This is the same predicate B7 needs. *Effort:* medium.

*Objection, and this one must be respected:* the blanket call is very likely deliberate insurance, because `mesh_generate_flags` is a weak corner-only test that can miss curved and user-warped elements. Removing the call **before** hardening the predicate would expose XSMM builds on curved meshes to a wrong Helmholtz operator — worse than the current cost. The hardening has to land first or in the same change. This is a design and reproducibility issue, never a wrong-answer bug in its own right.

---

## Checked and not a problem

Recorded so the next reader does not re-derive them.

**`ortho` is correct, and it looks wrong.** `src/math/operators.f90:350-368` removes the constant mode by subtracting `glsum(x)/glb_n_points`, where `glb_n_points` counts GLL points **including duplicated copies** at shared dofs. That looks like the wrong denominator, since solvability of the singular pure-Neumann system requires the sum over *unique* dofs to vanish. It is right, because of where it is applied. At `fluid_pnpn.f90:846-852` it runs on the **unassembled** residual, *before* the gather-scatter. There `Σ_local r = 1ᵀQᵀr = Σ_unique b`, exactly the quantity that must vanish; and after the gather-scatter each unique dof has picked up `m_j·c`, so the total correction is `c·N_redundant = Σ_local r` and the projection is exact. **Both apparent mistakes — the pre-gather-scatter placement and the redundant count — are load-bearing.** Moving the call after the gather-scatter, or "fixing" the denominator, breaks it. (Verified the hard way: a test harness that applied `ortho` after the gather-scatter produced a right-hand side outside the operator's range.)

**`h2` is not zeroed in the device pressure residual**, unlike the CPU path, so it carries stale data from the previous velocity residual. Every consumer was checked: `ax_helm` gates the `h2` term on `ifh2` on all backends, and the preconditioners do not read it unguarded. Harmless — though the asymmetry with the CPU path is gratuitous.

**`phmg_resid_monitor` and `print_resid_info`** are unreferenced device-only debug helpers. On a CPU build they abort with an explicit diagnostic rather than reading null pointers, and on a device build the residual genuinely is formed on the device, so the numbers they print are correct. Not a defect.

**The `add4` family and the `tamg_print_*` helpers** are unreferenced by design — API symmetry and debug utilities in a live module. Reporting them would be churn.

---

## Method and caveats

Items were found by reading the source, then verified twice: once by the reader who found them, once independently by a second reader instructed to refute rather than confirm. The second pass materially changed several entries and eliminated others — D1's framing (a maintenance issue, not a live performance cost), B5's scope (`phmg` users, not every pressure solve), and E1's blast radius (libxsmm builds, not every backend) are all second-pass corrections.

**No GPU measurements were taken.** Performance items are stated as traffic counts, kernel-launch counts and structural redundancies, all of which are checkable by reading; where a speedup is implied it is a model. The two timing figures quoted (D1's quadratic scan, D2's 96.6 ns/point) are CPU microbenchmarks of loops transcribed verbatim from the source, not measurements of Neko itself.

Where an item names a default or an example, that was checked against the shipped case files rather than assumed.
