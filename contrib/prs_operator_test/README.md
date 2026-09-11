# Pn-Pn pressure operator: symmetry, definiteness and separability

Assembles the Neko `pnpn` pressure operator **exactly as the Krylov solver
applies it** and tests whether it is SPD, and whether it is separable on a
tensor-product (channel) mesh. Also does the same for the velocity Helmholtz
operator.

## What is assembled, and why it is the right object

Inside `gmres.f90:225-228` (and identically `cg.f90:207-209`) the operator is

    call Ax%compute(w, z, coef, msh, Xh)
    call gs_h%op(w, n, GS_OP_ADD)
    call bc_projector%apply(w, n)

`assemble.f90` runs a real case for a few steps, re-executes `prs_res%compute`
(the routine that sets `c_Xh%h1`, `c_Xh%h2` and `ifh2`, and therefore *defines*
the operator), then applies that same triple column by column.

The unique-dof numbering comes from the gather-scatter itself (`GS_OP_MIN` on
the local index), so it is the operator's own notion of dof identity rather
than an assumption. Because `coef%mult = 1/multiplicity`, summing
`u(rep(i))*Au(rep(i))` over unique dofs is *exactly* the `glsc3(.,.,coef%mult)`
inner product the Krylov solver minimises in — so "the matrix is symmetric"
means "the operator is self-adjoint in the solver's own norm".

Self-checks printed at run time: representative-copy consistency
(`max|w(i) - w(rep(l2g(i)))|`, must be 0, since `bc_projector%apply` is a plain
local index list with no gather-scatter propagation), and right-hand-side
consistency (`sum_unique(b)`, must be ~0 for the pure-Neumann case).

## Limitations

Serial, CPU backend, double precision only — enforced with `neko_error` at
startup. Dense assembly is skipped above 14000 unique dofs; the matrix-free
symmetry/definiteness checks still run.

## Build and run

    mpif90 -O2 -fallow-argument-mismatch -I<prefix>/include/neko \
        -o assemble assemble.f90 -L<prefix>/lib -lneko -ljsonfortran -llapack -lblas

    ./mkmesh.sh                   # writes the three meshes
    ./assemble channel.case       # case A: periodic x,z + no-slip walls -> pure Neumann
    python3 analyze.py           # symmetry, spectrum, null space
    python3 sep.py               # A vs Kx(x)My(x)Mz + Mx(x)Ky(x)Mz + Mx(x)My(x)Kz
    python3 fd.py                # direct fast-diagonalisation solve vs Neko's KSP
    python3 velsep.py            # velocity Helmholtz, with the real wall mask
    python3 circulant.py          # is a uniform periodic direction FFT-diagonalisable?

    ./assemble channel_order7.case  # case C: order 7, matrix-free checks only
    ./assemble outflow.case         # case B: with a Dirichlet pressure bc
    python3 mask_analysis.py        # where the asymmetry lives

## Results

Neko 1.99.9 (based on `ce9b260`), CPU, fp64, gfortran 13.3.

### Case A -- channel, pure Neumann pressure

Channel `2pi x 2 x pi`,
periodic in x and z, no-slip walls in y. Mesh **non-uniform in x** (widths
1.65/2.65/1.98) and tanh-stretched in y (heights ratio 8.1). Order 4 (`lx=5`),
54 elements, 3600 unique dofs.

Operator state read from the live objects: `h1 = 1.0` everywhere (`= 1/rho`),
`h2 = 0`, `ifh2 = F`, `prs_dirichlet = F`, pressure mask size 0.

| quantity | value |
| --- | --- |
| `max abs(A - A^T) / max abs(A)` | 1.25e-16 (eps = 2.22e-16) |
| `norm_F(A - A^T) / norm_F(A)` | 6.44e-17 |
| matrix-free `<u,Av>` vs `<v,Au>`, relative | 1.57e-16 |
| `abs(A*1)_inf` (constant mode) | 2.84e-14 |
| negative eigenvalues | 0 |
| zero eigenvalues | 1 |
| 2nd smallest / largest eigenvalue | 9.33e-03 / 1.07e+02 |
| `max abs(Im lambda) / max abs(Re lambda)` | 1.17e-17 |
| `max abs(A - A_separable) / max abs(A)` | 4.26e-14 |
| representative-copy consistency | 0.0 exactly |

i.e. symmetric positive semi-definite with a one-dimensional null space of
constants, and exactly equal to the separable Kronecker sum.

### Case C -- same geometry, larger and higher order

4 x 8 x 3 elements, non-uniform in x (ratio 2.6), y ratio 14.6, order 7
(`lx = 8`), 33516 unique dofs. Too large to assemble densely; matrix-free only:
relative asymmetry 3.45e-16, `<v,Av>` = 2.3e+04 > 0 on every sample,
`abs(A*1)_inf` = 1.24e-14, representative-copy discrepancy 0. The sample
vectors are randomly seeded, so the asymmetry figure varies run to run at the
1e-16 level.

### Case A -- direct solve

Direct separable (fast-diagonalisation) solve against Neko's own GMRES+hsmg,
same operator, same manufactured right-hand side:

| | residual `norm(Ax-b)/norm(b)` | error vs exact |
| --- | --- | --- |
| Neko GMRES + hsmg, 33 iterations | 1.31e-08 | 3.71e-09 |
| direct fast diagonalisation | 8.09e-14 | 8.05e-14 |
| agreement between the two | | 3.71e-09 |

Velocity Helmholtz (`h1 = mu = 3.571e-4`, `h2 = rho*bd/dt = 183.33`, both
constant), with the real per-component no-slip mask applied (288 masked rows =
the two wall planes exactly): symmetric to 1.19e-19, strictly positive definite
(min eigenvalue 6.35e-02, condition number 230), and equal to
`mu * Laplacian_sep + h2 * Mass_sep` to 2.85e-15.

`circulant.py`: on a **uniform** periodic element line the assembled 1-D
operator is block-circulant with block size `lx-1`, so an FFT across elements
block-diagonalises it to 1.2e-16 (control: non-uniform spacing gives 1.6e-01).

## Case B -- Dirichlet pressure boundary condition (`outflow.case`)

A different mesh: 3 x 4 x 3 elements, uniform in x and z, tanh-stretched over 4
wall-normal layers, x non-periodic with a prescribed-velocity inflow on one
face and an `outflow` (Dirichlet pressure) on the other. 2652 unique dofs.


`bcs_prs_projector%apply` zeroes the operator **output** only, so Neko applies
`M*A`, not `M*A*M`:

| quantity | value |
| --- | --- |
| total unique dofs | 2652 |
| `max abs(A - A^T) / max abs(A)` | 2.26e-02 (not round-off) |
| all-zero rows / all-zero columns | 204 / 0 |
| restricted to the 2448 unmasked dofs: symmetry | 1.54e-16 |
| restricted: min / max eigenvalue | 9.24e-04 / 3.21e+01 |
| restricted: negative / zero eigenvalues | 0 / 0 |
| `M A M` symmetry | 1.54e-16 |

The asymmetry lives entirely in the masked columns. Every Krylov vector is
masked (the initial residual at `fluid_pnpn.f90:857`, every operator output at
`gmres.f90:248`), so that subspace is `A`-invariant and the iteration only ever
sees the symmetric part — but this is preconditioner-dependent and is **not**
tested here.
