!> Assemble the Pn-Pn pressure operator exactly as the Krylov solver sees it.
program assemble_prs_operator
  use neko
  use num_types, only : i8
  use fluid_pnpn, only : fluid_pnpn_t
  use gs_ops, only : GS_OP_ADD, GS_OP_MIN
  use operators, only : ortho
  use krylov, only : ksp_monitor_t
  use num_types, only : dp
  use comm, only : pe_size
  use neko_config, only : NEKO_BCKND_DEVICE
  use vector_bc_projector, only : vector_bc_projector_components
  use scalar_bc_projector, only : scalar_bc_projector_t
  implicit none

  type(case_t), target :: C
  integer :: n, nglb, i, j, k, e, l, lx, lunit
  integer(kind=i8), allocatable :: gid(:)
  integer, allocatable :: l2g(:), rep(:)
  real(kind=rp), allocatable :: pin(:), w(:), A(:,:)
  real(kind=rp), allocatable :: gx(:), gy(:), gz(:), gmult(:)
  type(c_ptr) :: ev = C_NULL_PTR

  call neko_init(C)

  ! The harness labels dofs by rank-local index and writes one shared file, so
  ! it is serial-only.  It also reads and writes host arrays throughout, and
  ! encodes integer dof labels in reals, so it needs a CPU double-precision
  ! build.  Fail loudly rather than producing quietly wrong matrices.
  if (pe_size .gt. 1) call neko_error( &
       'assemble: serial only, run with a single MPI rank')
  if (NEKO_BCKND_DEVICE .eq. 1) call neko_error( &
       'assemble: CPU backend only, no host/device transfers are issued')
  if (rp .ne. dp) call neko_error( &
       'assemble: double precision build only (dof labels are held in reals)')

  call neko_solve(C)

  select type (fl => C%fluid)
  type is (fluid_pnpn_t)

     n = fl%dm_Xh%size()
     lx = fl%Xh%lx

     ! Re-run the real pressure residual: this is what sets h1/h2/ifh2 and
     ! therefore defines the operator the Krylov solver is handed.
     call fl%prs_res%compute(fl%p, fl%p_res, fl%u, fl%v, fl%w, &
          fl%u_e, fl%v_e, fl%w_e, fl%f_x, fl%f_y, fl%f_z, &
          fl%c_Xh, fl%gs_Xh, fl%bc_prs_surface, fl%bc_sym_surface, &
          fl%Ax_prs, fl%ext_bdf%diffusion_coeffs%x(1), &
          real(C%time%dt, kind=rp), fl%mu_tot, fl%rho, ev)

     write(*,*) '=== OPERATOR STATE AT THE PRESSURE SOLVE ==='
     write(*,*) 'h1  min/max   : ', minval(fl%c_Xh%h1), maxval(fl%c_Xh%h1)
     write(*,*) 'h2  min/max   : ', minval(fl%c_Xh%h2), maxval(fl%c_Xh%h2)
     write(*,*) 'ifh2          : ', fl%c_Xh%ifh2
     write(*,*) 'prs_dirichlet : ', fl%prs_dirichlet
     write(*,*) 'prs mask size : ', fl%bcs_prs_projector%dof_mask%size()
     write(*,*) 'lx/nelv/ndofl : ', lx, fl%msh%nelv, n
     write(*,*) 'rho / mu      : ', fl%rho%x(1,1,1,1), fl%mu_tot%x(1,1,1,1)

     ! Canonical unique-dof labelling taken straight from the gather-scatter:
     ! v(i) = i, then GS_OP_MIN leaves every copy of a shared dof carrying the
     ! smallest local index among its copies.  Distinct values = unique dofs.
     allocate(pin(n), w(n))
     do i = 1, n
        w(i) = real(i, kind=rp)
     end do
     call fl%gs_Xh%op(w, n, GS_OP_MIN)
     allocate(gid(n), l2g(n))
     do i = 1, n
        gid(i) = int(nint(w(i)), kind=i8)
     end do
     call build_unique(gid, n, l2g, nglb)
     allocate(rep(nglb)); rep = 0
     do i = 1, n
        if (rep(l2g(i)) .eq. 0) rep(l2g(i)) = i
     end do
     write(*,*) 'unique dofs   : ', nglb

     allocate(gx(nglb), gy(nglb), gz(nglb), gmult(nglb))
     l = 0
     do e = 1, fl%msh%nelv
        do k = 1, lx
           do j = 1, lx
              do i = 1, lx
                 l = l + 1
                 gx(l2g(l)) = fl%dm_Xh%x(i,j,k,e)
                 gy(l2g(l)) = fl%dm_Xh%y(i,j,k,e)
                 gz(l2g(l)) = fl%dm_Xh%z(i,j,k,e)
                 gmult(l2g(l)) = fl%c_Xh%mult(i,j,k,e)
              end do
           end do
        end do
     end do

     ! ================= MATRIX-FREE CHECKS (scale to any size) =================
     block
       integer, parameter :: NT = 8
       real(kind=rp), allocatable :: U(:,:), AU(:,:), gu(:)
       real(kind=rp) :: s1, s2, asym, nrm, quad, qmin, anorm
       integer :: a2, b2, t
       allocate(U(n, NT), AU(n, NT), gu(nglb))
       call random_seed()
       do t = 1, NT
          do i = 1, nglb
             call random_number(gu(i))
             gu(i) = gu(i) - 0.5_rp
          end do
          do i = 1, n
             U(i, t) = gu(l2g(i))          ! continuous by construction
          end do
          call fl%Ax_prs%compute(AU(:, t), U(:, t), fl%c_Xh, fl%msh, fl%Xh)
          call fl%gs_Xh%op(AU(:, t), n, GS_OP_ADD)
          call fl%bcs_prs_projector%apply(AU(:, t), n)
       end do
       write(*,*) '--- matrix-free symmetry:  <u,Av> vs <v,Au>  (global inner prod) ---'
       asym = 0.0_rp; nrm = 0.0_rp
       do a2 = 1, NT
          do b2 = 1, NT
             s1 = 0.0_rp; s2 = 0.0_rp
             do i = 1, nglb
                s1 = s1 + U(rep(i), a2) * AU(rep(i), b2)
                s2 = s2 + U(rep(i), b2) * AU(rep(i), a2)
             end do
             asym = max(asym, abs(s1 - s2))
             nrm = max(nrm, abs(s1))
          end do
       end do
       write(*,*) '  max |<u,Av>-<v,Au>|          :', asym
       write(*,*) '  relative to max |<u,Av>|     :', asym / nrm
       write(*,*) '--- matrix-free definiteness: <v,Av> for random continuous v ---'
       qmin = huge(qmin)
       do t = 1, NT
          quad = 0.0_rp
          do i = 1, nglb
             quad = quad + U(rep(i), t) * AU(rep(i), t)
          end do
          qmin = min(qmin, quad)
       end do
       write(*,*) '  min <v,Av> over samples      :', qmin
       ! constant vector
       do i = 1, n
          U(i,1) = 1.0_rp
       end do
       call fl%Ax_prs%compute(AU(:,1), U(:,1), fl%c_Xh, fl%msh, fl%Xh)
       call fl%gs_Xh%op(AU(:,1), n, GS_OP_ADD)
       call fl%bcs_prs_projector%apply(AU(:,1), n)
       anorm = 0.0_rp
       do i = 1, nglb
          anorm = max(anorm, abs(AU(rep(i),1)))
       end do
       write(*,*) '  ||A*1||_inf (constant mode)  :', anorm

       ! The representative-copy extraction A(i,j) = w(rep(i)) is only
       ! well-defined if the masked operator output is gs-continuous.  gs_op
       ! makes it continuous, but bcs_prs_projector%apply is a plain local
       ! index list with no gather-scatter propagation, so a boundary zone
       ! that masks a dof in one element but not in its neighbour would break
       ! this.  Measure it rather than assume it.
       anorm = 0.0_rp; quad = 0.0_rp
       do i = 1, n
          anorm = max(anorm, abs(AU(i,1) - AU(rep(l2g(i)),1)))
          quad = max(quad, abs(AU(i,1)))
       end do
       write(*,*) '--- representative-copy consistency (must be ~0) ---'
       write(*,*) '  max|w(i) - w(rep(l2g(i)))|   :', anorm
     end block

     if (nglb .gt. 14000) then
        write(*,*) 'skipping dense assembly (nglb too large)'
        call neko_finalize(C)
        stop
     end if
     allocate(A(nglb, nglb))
     do j = 1, nglb
        pin = 0.0_rp
        do i = 1, n
           if (l2g(i) .eq. j) pin(i) = 1.0_rp
        end do
        call fl%Ax_prs%compute(w, pin, fl%c_Xh, fl%msh, fl%Xh)
        call fl%gs_Xh%op(w, n, GS_OP_ADD)
        call fl%bcs_prs_projector%apply(w, n)
        do i = 1, nglb
           A(i, j) = w(rep(i))
        end do
     end do

     ! --- Controlled test: manufactured exact solution through Neko's own
     ! --- operator, then solved by Neko's own Krylov solver.
     block
       real(kind=rp), allocatable :: xex(:), b(:), gxex(:), gb(:), gksp(:)
       type(ksp_monitor_t) :: km
       real(kind=rp) :: cshift
       allocate(xex(n), b(n), gxex(nglb), gb(nglb), gksp(nglb))
       ! smooth, well-resolved manufactured pressure field (continuous by
       ! construction: it is a function of the coordinates only)
       do i = 1, n
          xex(i) = 0.0_rp
       end do
       l = 0
       do e = 1, fl%msh%nelv
          do k = 1, lx
             do j = 1, lx
                do i = 1, lx
                   l = l + 1
                   xex(l) = cos(fl%dm_Xh%x(i,j,k,e)) * &
                        sin(2.0_rp*fl%dm_Xh%z(i,j,k,e)) * &
                        (fl%dm_Xh%y(i,j,k,e)**2 - 1.0_rp) &
                        + 0.3_rp * sin(2.0_rp*fl%dm_Xh%x(i,j,k,e)) * &
                        cos(2.0_rp*fl%dm_Xh%z(i,j,k,e)) * fl%dm_Xh%y(i,j,k,e)
                end do
             end do
          end do
       end do
       ! Fix the free constant by de-meaning over the UNIQUE dofs.  (Neko's
       ! own ortho() divides by the redundant point count glb_n_points, which
       ! is the right thing for the residual it is applied to but is not a
       ! unique-dof mean; here we only need a canonical representative.)
       cshift = 0.0_rp
       do i = 1, nglb
          cshift = cshift + xex(rep(i))
       end do
       cshift = cshift / real(nglb, kind=rp)
       do i = 1, n
          xex(i) = xex(i) - cshift
       end do
       do i = 1, nglb
          gxex(i) = xex(rep(i))
       end do
       ! b = A x_exact, using exactly the operator the Krylov solver applies.
       ! NOTE: no ortho() here.  b is by construction in range(A), and Neko's
       ! ortho() applied AFTER the gather-scatter would subtract a
       ! multiplicity-weighted mean and push a consistent rhs out of range.
       call fl%Ax_prs%compute(b, xex, fl%c_Xh, fl%msh, fl%Xh)
       call fl%gs_Xh%op(b, n, GS_OP_ADD)
       call fl%bcs_prs_projector%apply(b, n)
       cshift = 0.0_rp
       do i = 1, nglb
          cshift = cshift + b(rep(i))
       end do
       write(*,*) 'rhs consistency  sum_unique(b) :', cshift
       do i = 1, n
          fl%p_res%x(i,1,1,1) = b(i)
       end do
       call fl%pc_prs%update()
       km = fl%ksp_prs%solve(fl%Ax_prs, fl%dp, fl%p_res%x, n, fl%c_Xh, &
            fl%bcs_prs_projector, fl%gs_Xh)
       write(*,*) 'KSP(manufactured): iters =', km%iter, &
            ' res start/final =', km%res_start, km%res_final
       do i = 1, nglb
          gb(i) = b(rep(i))
          gksp(i) = fl%dp%x(rep(i),1,1,1)
       end do
       open(newunit = lunit, file = 'rhs_sol.bin', form = 'unformatted', &
            access = 'stream', status = 'replace')
       write(lunit) nglb
       write(lunit) gxex, gb, gksp
       close(lunit)
       write(*,*) 'wrote rhs_sol.bin (x_exact, b, x_ksp)'
     end block

     ! ================= VELOCITY HELMHOLTZ OPERATOR ==========================
     block
       real(kind=rp), allocatable :: Av(:,:), pv(:), wv(:)
       integer :: lu2
       type(scalar_bc_projector_t), pointer :: bcx, bcy, bcz
       ! vel_res%compute sets h1 = mu, h2 = rho*bd/dt, ifh2 = .true.
       call fl%vel_res%compute(fl%Ax_vel, fl%u, fl%v, fl%w, &
            fl%u_res, fl%v_res, fl%w_res, fl%p, fl%f_x, fl%f_y, fl%f_z, &
            fl%c_Xh, fl%msh, fl%Xh, fl%mu_tot, fl%rho, &
            fl%ext_bdf%diffusion_coeffs%x(1), real(C%time%dt, kind=rp), n)
       write(*,*) '--- VELOCITY OPERATOR STATE ---'
       write(*,*) '(this is the SCALAR Helmholtz that Ax_vel%compute_vector'
       write(*,*) ' applies to each component in the no-model formulation;'
       write(*,*) ' the full velocity solve additionally applies rotate_cyc,'
       write(*,*) ' which is a no-op without cyclic bcs, and the per-component'
       write(*,*) ' Dirichlet mask, which is applied below.)'
       write(*,*) 'h1 min/max :', minval(fl%c_Xh%h1), maxval(fl%c_Xh%h1)
       write(*,*) 'h2 min/max :', minval(fl%c_Xh%h2), maxval(fl%c_Xh%h2)
       write(*,*) 'ifh2       :', fl%c_Xh%ifh2
       bcx => null(); bcy => null(); bcz => null()
       call vector_bc_projector_components(fl%bcs_vel_projector, bcx, bcy, bcz)
       if (associated(bcx)) write(*,*) 'x-velocity mask size :', &
            bcx%dof_mask%size()
       allocate(Av(nglb, nglb), pv(n), wv(n))
       do j = 1, nglb
          pv = 0.0_rp
          do i = 1, n
             if (l2g(i) .eq. j) pv(i) = 1.0_rp
          end do
          call fl%Ax_vel%compute(wv, pv, fl%c_Xh, fl%msh, fl%Xh)
          call fl%gs_Xh%op(wv, n, GS_OP_ADD)
          if (associated(bcx)) call bcx%apply(wv, n)
          do i = 1, nglb
             Av(i, j) = wv(rep(i))
          end do
       end do
       open(newunit = lu2, file = 'vel_op.bin', form = 'unformatted', &
            access = 'stream', status = 'replace')
       write(lu2) nglb
       write(lu2) Av
       write(lu2) fl%c_Xh%h1(1,1,1,1), fl%c_Xh%h2(1,1,1,1)
       close(lu2)
       write(*,*) 'wrote vel_op.bin'
       deallocate(Av, pv, wv)
     end block

     open(newunit = lunit, file = 'prs_op.bin', form = 'unformatted', &
          access = 'stream', status = 'replace')
     write(lunit) nglb
     write(lunit) A
     write(lunit) gx, gy, gz, gmult
     close(lunit)

     open(newunit = lunit, file = 'space1d.bin', form = 'unformatted', &
          access = 'stream', status = 'replace')
     write(lunit) lx, fl%msh%nelv
     write(lunit) fl%Xh%zg(:,1)
     write(lunit) fl%Xh%wx
     write(lunit) fl%Xh%dx
     close(lunit)
     write(*,*) 'wrote prs_op.bin and space1d.bin'

  class default
     call neko_error('not pnpn')
  end select

  call neko_finalize(C)

contains

  subroutine build_unique(g, m, map, nu)
    integer(kind=i8), intent(in) :: g(:)
    integer, intent(in) :: m
    integer, intent(inout) :: map(:)
    integer, intent(out) :: nu
    integer(kind=i8), allocatable :: s(:)
    integer, allocatable :: perm(:)
    integer :: ii
    allocate(s(m), perm(m))
    s = g(1:m)
    do ii = 1, m
       perm(ii) = ii
    end do
    call qsort_i8(s, perm, 1, m)
    nu = 0
    do ii = 1, m
       if (ii .eq. 1) then
          nu = nu + 1
       else if (s(ii) .ne. s(ii-1)) then
          nu = nu + 1
       end if
       map(perm(ii)) = nu
    end do
  end subroutine build_unique

  recursive subroutine qsort_i8(a, p, lo, hi)
    integer(kind=i8), intent(inout) :: a(:)
    integer, intent(inout) :: p(:)
    integer, intent(in) :: lo, hi
    integer :: ii, jj, tp
    integer(kind=i8) :: pv, ta
    if (lo .ge. hi) return
    pv = a((lo+hi)/2); ii = lo; jj = hi
    do while (ii .le. jj)
       do while (a(ii) .lt. pv)
          ii = ii + 1
       end do
       do while (a(jj) .gt. pv)
          jj = jj - 1
       end do
       if (ii .le. jj) then
          ta = a(ii); a(ii) = a(jj); a(jj) = ta
          tp = p(ii); p(ii) = p(jj); p(jj) = tp
          ii = ii + 1; jj = jj - 1
       end if
    end do
    call qsort_i8(a, p, lo, jj)
    call qsort_i8(a, p, ii, hi)
  end subroutine qsort_i8

end program assemble_prs_operator
