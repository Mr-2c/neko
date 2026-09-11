! Copyright (c) 2024-2026, The Neko Authors
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
!   * Redistributions of source code must retain the above copyright
!     notice, this list of conditions and the following disclaimer.
!
!   * Redistributions in binary form must reproduce the above
!     copyright notice, this list of conditions and the following
!     disclaimer in the documentation and/or other materials provided
!     with the distribution.
!
!   * Neither the name of the authors nor the names of its
!     contributors may be used to endorse or promote products derived
!     from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
!> Runtime statistics
!!
!! Collects wall-clock timings for the profiling regions declared through
!! `profiler_start_region`/`profiler_end_region`. Regions nest, so every
!! region is accounted for twice: an *inclusive* time (everything between
!! the start and the end of the region) and a *self* time (the inclusive
!! time minus the inclusive time of its direct children). Only the self
!! times form a partition of the run, so those are the numbers to use when
!! asking "where does the time go".
!!
!! ### Accelerator backends
!!
!! On the device backends, kernel launches are asynchronous. A plain host
!! timer around a region therefore measures the time spent *launching* the
!! work, not the time spent executing it, and the accumulated device work
!! is charged to whichever region happens to contain the next synchronising
!! operation (typically a `glsc3`-style reduction, a device-to-host copy,
!! or a gather-scatter). Set `case.runtime_statistics.sync_device` to
!! `true` to synchronise the device before each timestamp, which makes the
!! reported times reflect actual device execution. This serialises host and
!! device and removes any compute/communication overlap, so a synchronised
!! run is slower than a production run and should be used for attribution,
!! not for absolute performance numbers.
module runtime_stats
  use logger, only : neko_log, LOG_SIZE, NEKO_LOG_QUIET
  use stack, only : stack_r8_t
  use num_types, only : dp, i8
  use json_utils, only : json_get_or_default
  use json_module, only : json_file
  use utils, only : neko_error
  use comm, only : pe_rank, pe_size, NEKO_COMM
  use neko_config, only : NEKO_BCKND_DEVICE
  use device, only : device_sync
  use mpi_f08, only : MPI_Wtime, MPI_Allreduce, MPI_IN_PLACE, &
       MPI_DOUBLE_PRECISION, MPI_INTEGER, MPI_SUM, MPI_MAX
  implicit none
  private

  integer, parameter :: RT_STATS_MAX_REGIONS = 128
  integer, parameter :: RT_STATS_RESERVED_REGIONS = 64
  integer, parameter :: RT_STATS_MAX_NAME_LEN = 26
  !> Maximum nesting depth of profiling regions
  integer, parameter :: RT_STATS_MAX_DEPTH = 64

  !> Reserved id of the region delimiting one time step. Closing it rolls
  !! the per-step accumulators into the per-step sample history.
  integer, public, parameter :: RT_STATS_REGION_TIMESTEP = 23

  ! Ids 1..RT_STATS_RESERVED_REGIONS are assigned statically at the call
  ! site, which avoids a name lookup in the hot path. Ids above that range
  ! are handed out on demand to regions identified by name only. The
  ! reserved ids currently in use are:
  !
  !    1  Fluid                    26  Fluid_bc_apply
  !    2  <scalar name>            27  Ax_helm
  !    3  Pressure_solve           28  Ax_helm_vector
  !    4  Velocity_solve           29  Precon_apply
  !    5  gather_scatter           30  Dot_product
  !    6  gs_nbsend                31  Dot_product_many
  !    7  gs_nbwait                32  Krylov_ortho
  !    8  PHMG_solve, HSMG_solve   33  MPI_allreduce
  !    9  HSMG_schwarz             34  PHMG_smoother
  !   10  HSMG_coarse_grid         35  PHMG_residual
  !   11  HSMG_coarse-solve        36  PHMG_restrict
  !   12  gs_local                 37  PHMG_prolong
  !   13  gs_nbrecv                38  PHMG_coarse-solve
  !   14  gs_gather_shared         39  AMG_smoother
  !   15  gs_scatter_shared        40  AMG_matvec
  !   16  Project on               41  AMG_interp
  !   17  Project back             42  AMG_coarse_solve
  !   18  Pressure_residual        43  Krylov_update
  !   19  Velocity_residual        44  Pressure_pc_update
  !   20  <scalar>_residual        45  Velocity_pc_update
  !   21  <scalar>_solve           46  Opgrad
  !   22  Output controller        47  Cdtp
  !   23  Time-Step                48  Conv1
  !   24  Advection                49  Curl
  !   25  Fluid_source_terms       50  Interpolate

  !> Detail levels. A region declared at level @a l is only measured when
  !! `case.runtime_statistics.detail_level` is at least @a l.
  !> Top-level breakdown of a time step (the default).
  integer, public, parameter :: RT_LVL_BASIC = 1
  !> Internals of the Krylov solvers, preconditioners and multigrid cycles.
  integer, public, parameter :: RT_LVL_SOLVER = 2
  !> Individual operators and kernel groups.
  integer, public, parameter :: RT_LVL_KERNEL = 3

  type :: runtime_stats_t
     private
     !> Name of measured region
     character(len=RT_STATS_MAX_NAME_LEN), allocatable :: rt_stats_id(:)
     !> Detail level a region was declared at
     integer, allocatable :: region_level(:)
     !> Per-time-step inclusive time for each measured region
     type(stack_r8_t), allocatable :: elapsed_time(:)
     !> Inclusive time accumulated over the whole run
     real(kind=dp), allocatable :: total_time(:)
     !> Inclusive time of the direct children of each region, used to
     !! derive the self (exclusive) time
     real(kind=dp), allocatable :: child_time(:)
     !> Inclusive time accumulated in the current time step
     real(kind=dp), allocatable :: step_time(:)
     !> Number of times each region has been entered
     integer(kind=i8), allocatable :: n_calls(:)
     !> Ids of the currently open regions, outermost first
     integer :: open_id(RT_STATS_MAX_DEPTH) = 0
     !> Timestamps of the currently open regions
     real(kind=dp) :: open_start(RT_STATS_MAX_DEPTH) = 0.0_dp
     !> Number of currently open regions
     integer :: depth = 0
     !> Number of completed time steps
     integer :: nsteps = 0
     !> Regions declared above this level are not measured
     integer :: detail_level = RT_LVL_BASIC
     logical :: enabled = .false.
     logical :: output_profile = .false.
     !> Synchronise the device before every timestamp
     logical :: sync_device = .false.
   contains
     procedure, public, pass(this) :: init => runtime_stats_init
     procedure, public, pass(this) :: free => runtime_stats_free
     procedure, public, pass(this) :: start_region => runtime_stats_start_region
     procedure, public, pass(this) :: end_region => runtime_stats_end_region
     procedure, public, pass(this) :: report => runtime_stats_report

     procedure, pass(this) :: find_region_id => runtime_stats_find_region_id
     procedure, pass(this) :: end_step => runtime_stats_end_step
     procedure, pass(this) :: write_summary => runtime_stats_write_summary
     procedure, pass(this) :: write_timeline => runtime_stats_write_timeline
  end type runtime_stats_t

  type(runtime_stats_t), public :: neko_rt_stats

contains

  !> Initialise runtime statistics
  subroutine runtime_stats_init(this, params)
    class(runtime_stats_t), intent(inout) :: this
    type(json_file), intent(inout) :: params
    integer :: i

    call this%free()

    call json_get_or_default(params, 'case.runtime_statistics.enabled', &
         this%enabled, .false.)
    call json_get_or_default(params, &
         'case.runtime_statistics.output_profile', &
         this%output_profile, .false.)
    call json_get_or_default(params, &
         'case.runtime_statistics.detail_level', &
         this%detail_level, RT_LVL_BASIC)
    call json_get_or_default(params, &
         'case.runtime_statistics.sync_device', &
         this%sync_device, .false.)

    if (this%detail_level .lt. RT_LVL_BASIC .or. &
         this%detail_level .gt. RT_LVL_KERNEL) then
       call neko_error('Invalid runtime statistics detail level')
    end if

    ! Synchronising is only meaningful, and only safe, on a device backend
    this%sync_device = this%sync_device .and. (NEKO_BCKND_DEVICE .eq. 1)

    if (this%enabled) then

       allocate(this%rt_stats_id(RT_STATS_MAX_REGIONS))
       allocate(this%region_level(RT_STATS_MAX_REGIONS))
       allocate(this%total_time(RT_STATS_MAX_REGIONS))
       allocate(this%child_time(RT_STATS_MAX_REGIONS))
       allocate(this%step_time(RT_STATS_MAX_REGIONS))
       allocate(this%n_calls(RT_STATS_MAX_REGIONS))

       this%rt_stats_id = ''
       this%region_level = RT_LVL_BASIC
       this%total_time = 0.0_dp
       this%child_time = 0.0_dp
       this%step_time = 0.0_dp
       this%n_calls = 0_i8

       allocate(this%elapsed_time(RT_STATS_MAX_REGIONS))
       do i = 1, RT_STATS_MAX_REGIONS
          call this%elapsed_time(i)%init()
       end do

       this%depth = 0
       this%nsteps = 0

    end if

  end subroutine runtime_stats_init

  !> Destroy runtime statistics
  subroutine runtime_stats_free(this)
    class(runtime_stats_t), intent(inout) :: this
    integer :: i

    if (allocated(this%rt_stats_id)) then
       deallocate(this%rt_stats_id)
    end if

    if (allocated(this%region_level)) then
       deallocate(this%region_level)
    end if

    if (allocated(this%total_time)) then
       deallocate(this%total_time)
    end if

    if (allocated(this%child_time)) then
       deallocate(this%child_time)
    end if

    if (allocated(this%step_time)) then
       deallocate(this%step_time)
    end if

    if (allocated(this%n_calls)) then
       deallocate(this%n_calls)
    end if

    if (allocated(this%elapsed_time)) then
       do i = 1, size(this%elapsed_time)
          call this%elapsed_time(i)%free()
       end do
       deallocate(this%elapsed_time)
    end if

    this%depth = 0
    this%nsteps = 0

  end subroutine runtime_stats_free

  !> Start measuring time for the region
  !! named @a name with id @a region_id
  !! @param name Name of the region.
  !! @param region_id Optional id of the region, avoids a name lookup.
  !! @param level Optional detail level the region is declared at.
  subroutine runtime_stats_start_region(this, name, region_id, level)
    class(runtime_stats_t), intent(inout) :: this
    character(len=*), intent(in) :: name
    integer, optional, intent(in) :: region_id
    integer, optional, intent(in) :: level
    character(len=RT_STATS_MAX_NAME_LEN) :: region_name
    integer :: id, lvl

    if (.not. this%enabled) return

    lvl = RT_LVL_BASIC
    if (present(level)) lvl = level
    if (lvl .gt. this%detail_level) return

    ! Region names longer than RT_STATS_MAX_NAME_LEN are stored truncated,
    ! so compare against the truncated form rather than the argument.
    region_name = name

    if (present(region_id)) then
       id = region_id
    else
       call this%find_region_id(region_name, id)
    end if

    if (id .gt. 0 .and. id .le. RT_STATS_MAX_REGIONS) then
       if (len_trim(this%rt_stats_id(id)) .eq. 0) then
          this%rt_stats_id(id) = region_name
          this%region_level(id) = lvl
       else if (this%rt_stats_id(id) .ne. region_name) then
          call neko_error('Profile region renamed')
       end if
    else
       call neko_error('Invalid profiling region id')
    end if

    if (this%depth .ge. RT_STATS_MAX_DEPTH) then
       call neko_error('Profiling regions nested too deeply')
    end if

    if (this%sync_device) call device_sync()

    this%depth = this%depth + 1
    this%open_id(this%depth) = id
    this%open_start(this%depth) = MPI_Wtime()

  end subroutine runtime_stats_start_region

  !> Compute elapsed time for the current region
  !! @param name Optional name of the region to close.
  !! @param region_id Optional id of the region to close.
  !! @param level Optional detail level the region is declared at.
  subroutine runtime_stats_end_region(this, name, region_id, level)
    class(runtime_stats_t), intent(inout) :: this
    character(len=*), optional, intent(in) :: name
    integer, optional, intent(in) :: region_id
    integer, optional, intent(in) :: level
    real(kind=dp) :: end_time, elapsed_time
    character(len=1024) :: error_msg
    character(len=RT_STATS_MAX_NAME_LEN) :: region_name
    integer :: id, open_region, lvl

    if (.not. this%enabled) return

    lvl = RT_LVL_BASIC
    if (present(level)) lvl = level
    if (lvl .gt. this%detail_level) return

    if (this%sync_device) call device_sync()
    end_time = MPI_Wtime()

    if (this%depth .le. 0) then
       call neko_error('Invalid profiling region closed')
    end if

    open_region = this%open_id(this%depth)

    ! If we are given a name, check it matches the region being closed
    if (present(name)) then
       region_name = name
       if (present(region_id)) then
          id = region_id
       else
          call this%find_region_id(region_name, id)
       end if

       if (this%rt_stats_id(id) .ne. region_name) then
          write(error_msg, '(A,I0,A,A,A)') 'Invalid profiler region closed (', &
               id, ', expected: ', trim(this%rt_stats_id(id)), ')'
          call neko_error(trim(error_msg))

       else if (open_region .ne. id) then

          write(error_msg, '(A,A,A,A,A)') 'Invalid profiler region closed (', &
               trim(this%rt_stats_id(open_region)), ', expected: ', &
               trim(this%rt_stats_id(id)), ')'
          call neko_error(trim(error_msg))
       end if
    end if

    elapsed_time = end_time - this%open_start(this%depth)

    this%total_time(open_region) = this%total_time(open_region) + elapsed_time
    this%step_time(open_region) = this%step_time(open_region) + elapsed_time
    this%n_calls(open_region) = this%n_calls(open_region) + 1_i8

    this%depth = this%depth - 1

    ! Charge the time to the enclosing region as well, so that its self
    ! time can be recovered as inclusive time minus the time of its
    ! direct children.
    if (this%depth .ge. 1) then
       this%child_time(this%open_id(this%depth)) = &
            this%child_time(this%open_id(this%depth)) + elapsed_time
    end if

    if (open_region .eq. RT_STATS_REGION_TIMESTEP) call this%end_step()

  end subroutine runtime_stats_end_region

  !> Roll the per-time-step accumulators into the per-step history
  !!
  !! Every slot gets a sample, including the ones no region has claimed
  !! yet, so that row @a j of the history is always time step @a j even for
  !! a region that is first entered part-way into the run.
  subroutine runtime_stats_end_step(this)
    class(runtime_stats_t), intent(inout) :: this
    integer :: i

    do i = 1, RT_STATS_MAX_REGIONS
       call this%elapsed_time(i)%push(this%step_time(i))
       this%step_time(i) = 0.0_dp
    end do

    this%nsteps = this%nsteps + 1

  end subroutine runtime_stats_end_step

  !> Report runtime statistics for all recorded regions
  subroutine runtime_stats_report(this)
    class(runtime_stats_t), intent(inout) :: this
    character(len=LOG_SIZE) :: log_buf
    real(kind=dp) :: ref, pct, per_call, imbalance
    real(kind=dp), allocatable :: incl_avg(:), incl_max(:), self_avg(:)
    integer(kind=i8), allocatable :: calls(:)
    integer :: i, nregions

    if (.not. this%enabled) return

    nregions = RT_STATS_MAX_REGIONS
    allocate(incl_avg(nregions), incl_max(nregions), self_avg(nregions))
    allocate(calls(nregions))

    do i = 1, nregions
       incl_avg(i) = this%total_time(i)
       self_avg(i) = max(this%total_time(i) - this%child_time(i), 0.0_dp)
       calls(i) = this%n_calls(i)
    end do
    incl_max = incl_avg

    call MPI_Allreduce(MPI_IN_PLACE, incl_avg, nregions, &
         MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM)
    call MPI_Allreduce(MPI_IN_PLACE, self_avg, nregions, &
         MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM)
    call MPI_Allreduce(MPI_IN_PLACE, incl_max, nregions, &
         MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM)
    incl_avg = incl_avg / pe_size
    self_avg = self_avg / pe_size

    ! Percentages are relative to the time-step region when it is present,
    ! otherwise to the largest inclusive time recorded.
    ref = incl_avg(RT_STATS_REGION_TIMESTEP)
    if (ref .le. 0.0_dp) ref = maxval(incl_avg)
    if (ref .le. 0.0_dp) ref = 1.0_dp

    call neko_log%section('Runtime statistics', NEKO_LOG_QUIET)
    call neko_log%newline(NEKO_LOG_QUIET)

    if (this%sync_device) then
       call neko_log%message('Device synchronised at every region boundary; '&
       &// 'timings reflect', NEKO_LOG_QUIET)
       call neko_log%message('device execution but the run is serialised.', &
            NEKO_LOG_QUIET)
    else if (NEKO_BCKND_DEVICE .eq. 1) then
       call neko_log%message('Host timings on an asynchronous backend: '&
       &// 'device work is charged to', NEKO_LOG_QUIET)
       call neko_log%message('the region holding the next synchronising '&
       &// 'call. Set sync_device to', NEKO_LOG_QUIET)
       call neko_log%message('true for per-region attribution.', &
            NEKO_LOG_QUIET)
    end if
    call neko_log%newline(NEKO_LOG_QUIET)

    write(log_buf, '(A26,1x,A10,1x,A10,1x,A10,1x,A6,1x,A9)') &
         'Region', 'Calls/step', 'Total [s]', 'Self [s]', 'Self %', 'us/call'
    call neko_log%message(log_buf, NEKO_LOG_QUIET)
    write(log_buf, '(A)') repeat('-', 76)
    call neko_log%message(log_buf, NEKO_LOG_QUIET)

    do i = 1, nregions
       if (len_trim(this%rt_stats_id(i)) .gt. 0) then

          pct = 100.0_dp * self_avg(i) / ref
          if (calls(i) .gt. 0_i8) then
             per_call = 1.0e6_dp * incl_avg(i) / real(calls(i), dp)
          else
             per_call = 0.0_dp
          end if

          write(log_buf, &
               '(A26,1x,F10.1,1x,ES10.3,1x,ES10.3,1x,F6.2,1x,ES9.2)') &
               this%rt_stats_id(i), &
               real(calls(i), dp) / real(max(this%nsteps, 1), dp), &
               incl_avg(i), self_avg(i), pct, per_call
          call neko_log%message(log_buf, NEKO_LOG_QUIET)
       end if
    end do

    call neko_log%newline(NEKO_LOG_QUIET)

    ! Load imbalance across ranks. A region whose slowest rank spends much
    ! more time than the average is either unevenly partitioned or is
    ! absorbing the wait of a collective issued elsewhere.
    if (pe_size .gt. 1) then
       write(log_buf, '(A26,1x,A12,1x,A12,1x,A10)') &
            'Region', 'Avg [s]', 'Max [s]', 'Max/Avg'
       call neko_log%message(log_buf, NEKO_LOG_QUIET)
       write(log_buf, '(A)') repeat('-', 63)
       call neko_log%message(log_buf, NEKO_LOG_QUIET)
       do i = 1, nregions
          if (len_trim(this%rt_stats_id(i)) .gt. 0 .and. &
               incl_avg(i) .gt. 0.0_dp) then
             imbalance = incl_max(i) / incl_avg(i)
             write(log_buf, '(A26,1x,ES12.4,1x,ES12.4,1x,F10.3)') &
                  this%rt_stats_id(i), incl_avg(i), incl_max(i), imbalance
             call neko_log%message(log_buf, NEKO_LOG_QUIET)
          end if
       end do
       call neko_log%newline(NEKO_LOG_QUIET)
    end if

    if (pe_rank .eq. 0) then
       call this%write_summary(incl_avg, incl_max, self_avg, calls, ref)
    end if

    if (this%output_profile) call this%write_timeline()

    call neko_log%end_section()

    deallocate(incl_avg, incl_max, self_avg, calls)

  end subroutine runtime_stats_report

  !> Write the aggregated per-region table as a CSV file
  !! @param incl_avg Inclusive time averaged over the ranks.
  !! @param incl_max Inclusive time of the slowest rank.
  !! @param self_avg Self time averaged over the ranks.
  !! @param calls Number of times each region was entered on this rank.
  !! @param ref Reference time the percentages are taken against.
  subroutine runtime_stats_write_summary(this, incl_avg, incl_max, self_avg, &
       calls, ref)
    class(runtime_stats_t), intent(inout) :: this
    real(kind=dp), intent(in) :: incl_avg(:), incl_max(:), self_avg(:)
    integer(kind=i8), intent(in) :: calls(:)
    real(kind=dp), intent(in) :: ref
    integer :: i, unit, ierr
    real(kind=dp) :: nsteps

    open(newunit = unit, file = 'profile_summary.csv', status = 'replace', &
         action = 'write', iostat = ierr)
    if (ierr .ne. 0) return

    nsteps = real(max(this%nsteps, 1), dp)

    write(unit, '(A)') 'region,level,calls,calls_per_step,incl_total_s,'&
    &//'incl_per_step_s,self_total_s,self_per_step_s,self_percent,'&
    &//'us_per_call,incl_max_s,imbalance'

    do i = 1, size(incl_avg)
       if (len_trim(this%rt_stats_id(i)) .eq. 0) cycle
       write(unit, '(A,",",I0,",",I0,9(",",E16.8))') &
            trim(this%rt_stats_id(i)), &
            this%region_level(i), &
            calls(i), &
            real(calls(i), dp) / nsteps, &
            incl_avg(i), &
            incl_avg(i) / nsteps, &
            self_avg(i), &
            self_avg(i) / nsteps, &
            100.0_dp * self_avg(i) / ref, &
            merge(1.0e6_dp * incl_avg(i) / real(max(calls(i), 1_i8), dp), &
            0.0_dp, calls(i) .gt. 0_i8), &
            incl_max(i), &
            merge(incl_max(i) / incl_avg(i), 0.0_dp, incl_avg(i) .gt. 0.0_dp)
    end do

    close(unit)

  end subroutine runtime_stats_write_summary

  !> Write the per-time-step inclusive time of every region as a CSV file
  !!
  !! One row per time step, one column per region, holding the inclusive
  !! time averaged over the ranks. Use it to separate a genuine cost from a
  !! start-up transient, and to see how much a region varies from step to
  !! step.
  subroutine runtime_stats_write_timeline(this)
    class(runtime_stats_t), intent(inout) :: this
    real(kind=dp), allocatable :: samples(:,:)
    character(len=1250) :: hdr
    integer, allocatable :: cols(:)
    integer :: i, j, nsamples, nrows, ncols, unit, ierr

    ! The reduction below spans every region slot and a row count agreed by
    ! all ranks, so that a rank which never entered a given region still
    ! takes part in the same collective.
    nrows = 0
    do i = 1, RT_STATS_MAX_REGIONS
       nrows = max(nrows, this%elapsed_time(i)%size())
    end do
    call MPI_Allreduce(MPI_IN_PLACE, nrows, 1, MPI_INTEGER, MPI_MAX, &
         NEKO_COMM)

    if (nrows .eq. 0) return

    allocate(samples(nrows, RT_STATS_MAX_REGIONS))
    samples = 0.0_dp

    do i = 1, RT_STATS_MAX_REGIONS
       nsamples = this%elapsed_time(i)%size()
       if (nsamples .gt. 0) then
          select type (region_sample => this%elapsed_time(i)%data)
          type is (double precision)
             samples(1:nsamples, i) = region_sample(1:nsamples)
          end select
       end if
    end do

    call MPI_Allreduce(MPI_IN_PLACE, samples, nrows * RT_STATS_MAX_REGIONS, &
         MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM)
    samples = samples / pe_size

    if (pe_rank .eq. 0) then
       ! Only the regions this rank knows about get a column, so that the
       ! header and the rows stay in step.
       allocate(cols(RT_STATS_MAX_REGIONS))
       hdr = ''
       ncols = 0
       do i = 1, RT_STATS_MAX_REGIONS
          if (len_trim(this%rt_stats_id(i)) .eq. 0) cycle
          ncols = ncols + 1
          cols(ncols) = i
          if (ncols .eq. 1) then
             hdr = trim(this%rt_stats_id(i))
          else
             hdr = trim(hdr) // ',' // trim(this%rt_stats_id(i))
          end if
       end do

       if (ncols .gt. 0) then
          open(newunit = unit, file = 'profile.csv', status = 'replace', &
               action = 'write', iostat = ierr)
          if (ierr .eq. 0) then
             write(unit, '(A)') trim(hdr)
             do j = 1, nrows
                write(unit, '(*(E16.8,:,","))') &
                     (samples(j, cols(i)), i = 1, ncols)
             end do
             close(unit)
          end if
       end if
       deallocate(cols)
    end if

    deallocate(samples)

  end subroutine runtime_stats_write_timeline

  !> Find or allocate a region id for the named region @a name
  subroutine runtime_stats_find_region_id(this, name, region_id)
    class(runtime_stats_t), intent(inout) :: this
    character(len=*), intent(in) :: name
    integer, intent(out) :: region_id
    integer :: i

    region_id = -1

    if (.not. this%enabled) return

    ! Look for the region name first
    do i = RT_STATS_RESERVED_REGIONS + 1, RT_STATS_MAX_REGIONS
       if (this%rt_stats_id(i) .eq. name) then
          region_id = i
          exit
       end if
    end do

    ! If found, return
    if (region_id .ne. -1) return

    ! Otherwise, look for an empty slot
    do i = RT_STATS_RESERVED_REGIONS + 1, RT_STATS_MAX_REGIONS
       if (len_trim(this%rt_stats_id(i)) .eq. 0) then
          region_id = i
          exit
       end if
    end do

    if (region_id .eq. -1) then
       call neko_error('Not enough profiling regions available')
    end if

  end subroutine runtime_stats_find_region_id

end module runtime_stats
