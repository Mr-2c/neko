# Pn-Pn pressure operator: symmetry, definiteness and separability test

Assembles the pressure operator **exactly as the Krylov solver applies it** and
checks whether it is SPD and whether it is separable on a tensor-product
(channel) mesh.

`assemble.f90` runs a real `pnpn` case for a few steps, then re-executes
`prs_res%compute` -- the routine that sets `c_Xh%h1`, `c_Xh%h2` and `ifh2`, and
therefore *defines* the operator -- and assembles

    A = bcs_prs_projector%apply( gs_Xh%op( Ax_prs%compute(p), GS_OP_ADD ) )

column by column in the global unique-dof basis.  The unique-dof numbering is
taken from the gather-scatter itself (`GS_OP_MIN` on the local index), so it is
consistent with the operator by construction rather than by assumption.

It also runs scalable matrix-free checks (`<u,Av>` vs `<v,Au>`, `<v,Av>`,
`A*1`), a manufactured-solution comparison against Neko's own Krylov solve,
and assembles the velocity Helmholtz operator.

## Build

    mpif90 -O2 -fallow-argument-mismatch -I<prefix>/include/neko \
        -o assemble assemble.f90 -L<prefix>/lib -lneko -ljsonfortran -llapack -lblas

## Mesh

    # stretched wall-normal distribution, optionally non-uniform streamwise
    python3 -c "import numpy as np; ny=6; g=2.2; \
      eta=np.arange(ny+1)/ny; y=np.tanh(g*(2*eta-1))/np.tanh(g); \
      open('disty.csv','w').write(','.join(f'{v:.16e}' for v in y))"
    genmeshbox 0 6.283185307179586 -1 1 0 3.141592653589793 3 6 3 \
        .true. .false. .true. uniform disty.csv uniform

## Run

    ./assemble channel.case      # periodic x,z + walls in y  -> pure Neumann
    ./assemble outflow.case      # with a Dirichlet pressure bc
    python3 analyze.py           # symmetry, spectrum, null space
    python3 sep.py               # A vs Kx(x)My(x)Mz + Mx(x)Ky(x)Mz + Mx(x)My(x)Kz
    python3 fd.py                # direct fast-diagonalisation solve vs Neko's KSP

## Results (Neko 1.99.9, CPU backend, double precision)

Channel, periodic x/z, no-slip walls, mesh non-uniform in x (1.6x) and
tanh-stretched in y (8.1x), order 4, 54 elements, 3600 unique dofs:

| quantity | value |
| --- | --- |
| `max abs(A - A^T) / max abs(A)` | 1.3e-16 |
| `abs(A*1)` (constant mode) | 2.8e-14 |
| number of negative eigenvalues | 0 |
| number of zero eigenvalues | 1 |
| `max abs(Im lambda) / max abs(Re lambda)` | 1.2e-17 |
| `abs(A - A_separable) / max abs(A)` | 4.3e-14 |
| `A` after 4 steps vs after 9 steps | bit-identical |

i.e. symmetric positive semi-definite with a one-dimensional null space of
constants, and exactly equal to the separable Kronecker sum.

With a Dirichlet pressure bc (`outflow.case`) the mask is applied to the rows
only, so the matrix handed to the Krylov solver is genuinely non-symmetric
(2.3e-2 relative).  Restricted to the unmasked dofs it is symmetric to 1.5e-16
and strictly positive definite.
