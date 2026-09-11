#!/usr/bin/env python3
"""Turn a Neko ``profile_summary.csv`` into a performance breakdown.

Neko writes ``profile_summary.csv`` when ``case.runtime_statistics.enabled``
is set. This script ranks the regions by self time and, for the regions whose
memory traffic and floating-point work are known analytically, reports the
achieved bandwidth, the achieved flop rate and the arithmetic intensity, so
that each region can be placed against the machine balance of the device it
ran on.

The roofline numbers are only as good as the cost models below, which assume
that each kernel streams its operands from device memory once and keeps the
tensor-product intermediates in registers or shared memory. They are meant
for deciding whether a region is close to a hardware limit, not for
reporting a peak.

The models also assume that one entry into a region covers the whole local
mesh, which is what the device backends do. The generic CPU backend runs the
dealiased advection one element at a time, so profiling that backend needs
``--per-element Opgrad,Interpolate``; without it those two regions are
credited with a whole mesh per call and their rates come out a factor of
``elements / rank`` too high.

Usage:

    neko_profile_report.py profile_summary.csv --lx 8 --elements 216000 \\
        --ranks 8 --peak-bandwidth 3350 --peak-flops 34000

``--elements`` is the total number of spectral elements in the mesh and
``--ranks`` the number of MPI ranks the profiled run used, so that the
per-rank problem size is ``elements / ranks``. The peaks are per device, in
GB/s and GFLOP/s; the defaults describe no particular machine, so pass the
numbers for the device you profiled on.
"""

import argparse
import csv
import math
import sys

#: Regions that are pure communication. Their cost is set by message rate,
#: latency and link bandwidth rather than by anything the device computes.
COMM_REGIONS = {
    "gs_nbsend",
    "gs_nbrecv",
    "gs_nbwait",
    "MPI_allreduce",
}

#: Regions that only aggregate their children and hold no work of their own.
CONTAINER_REGIONS = {
    "Time-Step",
    "Fluid",
    "Fluid compressible",
    "Pressure_solve",
    "Velocity_solve",
    "PHMG_solve",
    "HSMG_solve",
    "Precon_apply",
    "PHMG_smoother",
    "PHMG_coarse-solve",
    "AMG_coarse_solve",
    "gather_scatter",
}


def _sem_helmholtz(lx, nelv, components):
    """Cost of a spectral-element Helmholtz operator application.

    Per element the kernel reads the input field, the seven metric arrays
    (h1 and the six symmetric geometric factors, shared between components)
    and writes the result. The work is six tensor contractions of
    ``lx**4`` multiply-adds plus the pointwise application of the metrics.
    """
    per_elem_words = components * 2 * lx**3 + 7 * lx**3
    flops = nelv * components * (12 * lx**4 + 17 * lx**3)
    return per_elem_words * nelv * 8, flops


def _streaming(arrays, flops_per_point):
    """Cost of a kernel that streams @a arrays vectors once."""

    def model(lx, nelv, _components=1):
        n = nelv * lx**3
        return n * arrays * 8, n * flops_per_point

    return model


def _tensor_interpolation(lx, nelv, _components=1):
    """Cost of a 3D tensor-product interpolation between two orders.

    Dealiasing maps ``lx`` to ``lxd = ceil(3 * lx / 2)``; the three
    directional passes cost ``2 * lx**a * lxd**b`` flops each.
    """
    # Only the dealiasing transfer follows the 3/2 rule; the multigrid
    # transfers move between two polynomial orders and are not modelled.
    lxd = math.ceil(1.5 * lx)
    flops = nelv * 2 * (lx**3 * lxd + lx**2 * lxd**2 + lx * lxd**3)
    words = nelv * (lx**3 + lxd**3)
    return words * 8, flops


#: Cost models keyed by region name. Each returns (bytes, flops) for one
#: entry into the region, given the per-rank element count.
COST_MODELS = {
    "Ax_helm": lambda lx, nelv: _sem_helmholtz(lx, nelv, 1),
    "Ax_helm_vector": lambda lx, nelv: _sem_helmholtz(lx, nelv, 3),
    # ur/us/ut from one field plus nine metric arrays, six contractions.
    "Opgrad": lambda lx, nelv: (
        nelv * (4 * lx**3 + 9 * lx**3) * 8,
        nelv * (6 * lx**4 + 15 * lx**3),
    ),
    "Cdtp": lambda lx, nelv: (
        nelv * (4 * lx**3 + 3 * lx**3) * 8,
        nelv * (6 * lx**4 + 6 * lx**3),
    ),
    "Curl": lambda lx, nelv: (
        nelv * (3 * lx**3 + 3 * lx**3 + 9 * lx**3) * 8,
        nelv * (18 * lx**4 + 30 * lx**3),
    ),
    "Interpolate": _tensor_interpolation,
    # Two vectors and a weight in, one scalar out: a multiply-add chain.
    "Dot_product": _streaming(3, 5),
    # y = y + a * x over three vectors.
    "Krylov_update": _streaming(3, 2),
}


def classify(row, lx, nelv, peak_bw, peak_flops, launch_us,
             per_element_regions=frozenset()):
    """Return (verdict, detail) for one region."""
    name = row["region"]
    calls = float(row["calls"])
    us_per_call = float(row["us_per_call"])
    imbalance = float(row["imbalance"])

    if name in COMM_REGIONS:
        detail = "imbalance %.2f" % imbalance
        if name == "MPI_allreduce":
            return "comm (global, latency)", detail
        return "comm (neighbour)", detail

    if name in CONTAINER_REGIONS:
        return "container", "aggregates nested regions"

    if calls == 0:
        return "unused", ""

    model = COST_MODELS.get(name)
    if model is None:
        if us_per_call < launch_us:
            return "latency?", "%.1f us/call, at launch-overhead scale" % us_per_call
        return "unmodelled", "%.1f us/call" % us_per_call

    # A region named by --per-element is entered once per element rather
    # than once per local mesh, so model a single element instead.
    per_element = name in per_element_regions
    nbytes, flops = model(lx, 1 if per_element else nelv)
    # Rates are taken against the self time: a modelled region that also
    # contains a gather-scatter or a reduction should not be credited with
    # the time its children spent.
    seconds = float(row["self_total_s"]) / calls
    if seconds <= 0.0:
        return "unused", ""

    gbs = nbytes / seconds / 1.0e9
    gflops = flops / seconds / 1.0e9
    intensity = flops / nbytes

    machine_balance = peak_flops / peak_bw
    frac_bw = gbs / peak_bw
    frac_flops = gflops / peak_flops

    detail = "%.4g GB/s (%.1f%% peak), %.4g GFLOP/s (%.1f%% peak), AI %.2f%s" % (
        gbs,
        100 * frac_bw,
        gflops,
        100 * frac_flops,
        intensity,
        " [per-element calls]" if per_element else "",
    )

    if max(frac_bw, frac_flops) > 1.05:
        return "model/peak mismatch", detail + (
            " -- above peak, so one call does not cover the whole local mesh"
            " (the CPU dealiasing loops per element) or --lx/--elements/the"
            " peaks are wrong")
    if us_per_call < launch_us and max(frac_bw, frac_flops) < 0.2:
        return "latency", detail
    if frac_bw >= 0.5:
        return "memory bound", detail
    if frac_flops >= 0.5:
        return "compute bound", detail
    if intensity < machine_balance:
        return "memory bound (below roofline)", detail
    return "compute bound (below roofline)", detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="profile_summary.csv written by Neko")
    ap.add_argument("--lx", type=int, required=True,
                    help="points per direction, i.e. polynomial_order + 1")
    ap.add_argument("--elements", type=int, required=True,
                    help="total number of spectral elements in the mesh")
    ap.add_argument("--ranks", type=int, default=1,
                    help="number of MPI ranks the profiled run used")
    ap.add_argument("--peak-bandwidth", type=float, default=2039.0,
                    help="device memory bandwidth in GB/s")
    ap.add_argument("--peak-flops", type=float, default=19500.0,
                    help="device double-precision peak in GFLOP/s")
    ap.add_argument("--launch-overhead", type=float, default=10.0,
                    help="kernel launch overhead in microseconds; regions "
                         "faster than this are latency limited")
    ap.add_argument("--top", type=int, default=0,
                    help="only show the N regions with the largest self time")
    ap.add_argument("--per-element", default="",
                    help="comma-separated regions that are entered once per "
                         "element rather than once per local mesh. The "
                         "generic CPU backend needs "
                         "--per-element Opgrad,Interpolate; the device "
                         "backends need nothing")
    args = ap.parse_args(argv)

    per_element = frozenset(r.strip() for r in args.per_element.split(",")
                            if r.strip())
    unknown = per_element - set(COST_MODELS)
    if unknown:
        ap.error("no cost model for %s" % ", ".join(sorted(unknown)))

    nelv = args.elements / args.ranks
    if nelv < 1:
        ap.error("fewer elements than ranks")

    with open(args.csv, newline="") as fh:
        rows = [
            {k: v.strip() for k, v in r.items()
             if k is not None and isinstance(v, str)}
            for r in csv.DictReader(fh)
        ]

    rows.sort(key=lambda r: float(r["self_total_s"]), reverse=True)
    if args.top:
        rows = rows[: args.top]

    step = next((float(r["incl_per_step_s"]) for r in rows
                 if r["region"] == "Time-Step"), None)

    print("Elements per rank : %.0f  (%.3g points)" % (nelv, nelv * args.lx**3))
    print("Machine balance   : %.1f flop/byte" %
          (args.peak_flops / args.peak_bandwidth))
    if step:
        print("Time per step     : %.4f s" % step)
    print()

    hdr = "%-26s %8s %7s %9s  %-28s %s" % (
        "Region", "self[%]", "calls", "us/call", "verdict", "detail")
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        verdict, detail = classify(r, args.lx, nelv, args.peak_bandwidth,
                                   args.peak_flops, args.launch_overhead,
                                   per_element)
        print("%-26s %8.2f %7.0f %9.1f  %-28s %s" % (
            r["region"][:26],
            float(r["self_percent"]),
            float(r["calls"]),
            float(r["us_per_call"]),
            verdict,
            detail,
        ))

    print()
    groups = {}
    for r in rows:
        verdict, _ = classify(r, args.lx, nelv, args.peak_bandwidth,
                              args.peak_flops, args.launch_overhead,
                              per_element)
        if verdict == "container":
            continue
        key = verdict.split(" (")[0]
        groups[key] = groups.get(key, 0.0) + float(r["self_percent"])
    print("Self time by class:")
    for k in sorted(groups, key=lambda x: -groups[x]):
        print("  %-30s %6.2f %%" % (k, groups[k]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
