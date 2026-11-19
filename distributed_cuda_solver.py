import os
import sys
import re
import json
import math
import time
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
KAPPA_MAX_THEORY = 0.24
CFL_SAFETY = 0.8


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], encoding="utf-8")
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)      # (source=-1, dest=+1) along axis 0
    nbr_left, nbr_right = cart_comm.Shift(1, 1)   # along axis 1

    return {
        "comm": comm, "cart_comm": cart_comm, "rank": rank, "size": size,
        "coords": tuple(coords), "gpu_uuid": gpu_uuid, "seed": instance_seed,
        "up": nbr_up, "down": nbr_down, "left": nbr_left, "right": nbr_right,
        "ny": GLOBAL_NY // 2, "nx": GLOBAL_NX // 2,
    }


# ==============================================================================
# TASK 2
# ==============================================================================
def initialize_subdomains(ctx):
    ny, nx = ctx["ny"], ctx["nx"]
    row_rank, col_rank = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)   # 1/513
    dy = 1.0 / (GLOBAL_NY + 1)

    # Same seed on every rank -> identical field parameters, no communication needed.
    rng = np.random.default_rng(ctx["seed"] % (2**63))
    ph1, ph2 = rng.uniform(0.0, 1.0, 2)
    centers = rng.uniform(0.2, 0.8, size=(3, 2))
    amps = rng.uniform(0.5, 1.0, size=3)
    src_c = rng.uniform(0.3, 0.7, size=2)

    gi = row_rank * ny + np.arange(ny + 2)   # global row index 0..513
    gj = col_rank * nx + np.arange(nx + 2)
    Y = (gi * dy)[:, None]
    X = (gj * dx)[None, :]

    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * (3.0 * X + ph1)) * np.cos(2 * np.pi * (2.0 * Y + ph2))
    u = np.zeros((ny + 2, nx + 2))
    for (cx, cy), a in zip(centers, amps):
        u += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.06 ** 2))
    S = 100.0 * np.exp(-((X - src_c[0]) ** 2 + (Y - src_c[1]) ** 2) / (2 * 0.05 ** 2))

    if ctx["up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx["down"] == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx["left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx["right"] == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return (np.ascontiguousarray(u), np.ascontiguousarray(kappa),
            np.ascontiguousarray(S), dx, dy)


# ==============================================================================
# TASK 3: pack / unpack kernels and async exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, ny, nx, s_up, s_down, s_left, s_right):
    idx = cuda.grid(1)
    if idx < nx:
        s_up[idx] = u[1, idx + 1]
        s_down[idx] = u[ny, idx + 1]
    if idx < ny:
        s_left[idx] = u[idx + 1, 1]
        s_right[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, ny, nx, r_up, r_down, r_left, r_right,
                        has_up, has_down, has_left, has_right):
    idx = cuda.grid(1)
    if idx < nx:
        if has_up:
            u[0, idx + 1] = r_up[idx]
        if has_down:
            u[ny + 1, idx + 1] = r_down[idx]
    if idx < ny:
        if has_left:
            u[idx + 1, 0] = r_left[idx]
        if has_right:
            u[idx + 1, nx + 1] = r_right[idx]


class HaloBuffers:
    def __init__(self, ny, nx):
        sizes = {"up": nx, "down": nx, "left": ny, "right": ny}
        self.d_send = {k: cuda.device_array(n, dtype=np.float64) for k, n in sizes.items()}
        self.d_recv = {k: cuda.device_array(n, dtype=np.float64) for k, n in sizes.items()}
        self.h_send = {k: cuda.pinned_array(n, dtype=np.float64) for k, n in sizes.items()}
        self.h_recv = {k: cuda.pinned_array(n, dtype=np.float64) for k, n in sizes.items()}
        for k in sizes:
            self.h_recv[k][:] = 0.0


# tag t: 0 = data travelling UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
def exchange_halos_async(d_u, bufs, ctx, stream_boundary):
    """Stage A (pack + async D2H on stream_boundary) and start of Stage B
    (sync, then post non-blocking Irecv/Isend). Returns the request list."""
    ny, nx = ctx["ny"], ctx["nx"]
    cart = ctx["cart_comm"]
    threads = 256
    blocks = (max(ny, nx) + threads - 1) // threads
    pack_halos_kernel[blocks, threads, stream_boundary](
        d_u, ny, nx, bufs.d_send["up"], bufs.d_send["down"],
        bufs.d_send["left"], bufs.d_send["right"])
    for k in ("up", "down", "left", "right"):
        bufs.d_send[k].copy_to_host(bufs.h_send[k], stream=stream_boundary)
    stream_boundary.synchronize()

    reqs = []
    # receives: what comes from my 'down' neighbour travelled UP (tag 0), etc.
    reqs.append(cart.Irecv([bufs.h_recv["down"], MPI.DOUBLE], source=ctx["down"], tag=0))
    reqs.append(cart.Irecv([bufs.h_recv["up"], MPI.DOUBLE], source=ctx["up"], tag=1))
    reqs.append(cart.Irecv([bufs.h_recv["right"], MPI.DOUBLE], source=ctx["right"], tag=2))
    reqs.append(cart.Irecv([bufs.h_recv["left"], MPI.DOUBLE], source=ctx["left"], tag=3))
    reqs.append(cart.Isend([bufs.h_send["up"], MPI.DOUBLE], dest=ctx["up"], tag=0))
    reqs.append(cart.Isend([bufs.h_send["down"], MPI.DOUBLE], dest=ctx["down"], tag=1))
    reqs.append(cart.Isend([bufs.h_send["left"], MPI.DOUBLE], dest=ctx["left"], tag=2))
    reqs.append(cart.Isend([bufs.h_send["right"], MPI.DOUBLE], dest=ctx["right"], tag=3))
    return reqs


def complete_halos_async(reqs, d_u, bufs, ctx, stream_boundary):
    """Stage C part 1: Waitall, async H2D, unpack into ghost cells."""
    ny, nx = ctx["ny"], ctx["nx"]
    MPI.Request.Waitall(reqs)
    for k in ("up", "down", "left", "right"):
        bufs.d_recv[k].copy_to_device(bufs.h_recv[k], stream=stream_boundary)
    threads = 256
    blocks = (max(ny, nx) + threads - 1) // threads
    unpack_halos_kernel[blocks, threads, stream_boundary](
        d_u, ny, nx, bufs.d_recv["up"], bufs.d_recv["down"],
        bufs.d_recv["left"], bufs.d_recv["right"],
        ctx["up"] != MPI.PROC_NULL, ctx["down"] != MPI.PROC_NULL,
        ctx["left"] != MPI.PROC_NULL, ctx["right"] != MPI.PROC_NULL)


# ==============================================================================
# TASK 4: tiled stencil + orchestrators
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, S, dt, inv_dx2, inv_dy2, lam,
                         r_start, r_end, c_start, c_end):
    """16x16 block, 18x18 shared tile. Updates cells with local indices
    r_start..r_end, c_start..c_end (inclusive)."""
    smem = cuda.shared.array(shape=(18, 18), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * 16 + ty
    j = c_start + cuda.blockIdx.x * 16 + tx
    H = u.shape[0]
    W = u.shape[1]

    in_arr = (i < H) and (j < W)
    if in_arr:
        smem[ty + 1, tx + 1] = u[i, j]
        if ty == 0 and i - 1 >= 0:
            smem[0, tx + 1] = u[i - 1, j]
        if ty == 15 and i + 1 < H:
            smem[17, tx + 1] = u[i + 1, j]
        if tx == 0 and j - 1 >= 0:
            smem[ty + 1, 0] = u[i, j - 1]
        if tx == 15 and j + 1 < W:
            smem[ty + 1, 17] = u[i, j + 1]
    cuda.syncthreads()   # reached by every thread (no early return above)

    if i >= r_start and i <= r_end and j >= c_start and j <= c_end:
        c = smem[ty + 1, tx + 1]
        kc = kappa[i, j]
        kn = 0.5 * (kappa[i + 1, j] + kc)
        ks = 0.5 * (kappa[i - 1, j] + kc)
        ke = 0.5 * (kappa[i, j + 1] + kc)
        kw = 0.5 * (kappa[i, j - 1] + kc)
        dyy = (kn * (smem[ty + 2, tx + 1] - c) - ks * (c - smem[ty, tx + 1])) * inv_dx2
        dxx = (ke * (smem[ty + 1, tx + 2] - c) - kw * (c - smem[ty + 1, tx])) * inv_dy2
        u_new[i, j] = c + dt * (dyy + dxx - lam * c * c * c + S[i, j])


def launch_stencil(d_u, d_un, d_k, d_S, p, r0, r1, c0, c1, stream):
    if r1 < r0 or c1 < c0:
        return
    grid = ((c1 - c0 + TILE_DIM) // TILE_DIM, (r1 - r0 + TILE_DIM) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        d_u, d_un, d_k, d_S, p["dt"], p["inv_dx2"], p["inv_dy2"], LAMBDA_DAMP,
        r0, r1, c0, c1)


def step_serialized(d_u, d_un, d_k, d_S, bufs, ctx, p, s_int, s_bnd):
    ny, nx = ctx["ny"], ctx["nx"]
    reqs = exchange_halos_async(d_u, bufs, ctx, s_bnd)
    complete_halos_async(reqs, d_u, bufs, ctx, s_bnd)
    launch_stencil(d_u, d_un, d_k, d_S, p, 1, ny, 1, nx, s_bnd)   # whole interior after exchange
    s_bnd.synchronize()
    s_int.synchronize()


def step_overlapped(d_u, d_un, d_k, d_S, bufs, ctx, p, s_int, s_bnd):
    ny, nx = ctx["ny"], ctx["nx"]
    # Stage A + B: pack, D2H, sync, post Irecv/Isend
    reqs = exchange_halos_async(d_u, bufs, ctx, s_bnd)
    # deep interior on stream_interior, concurrent with MPI traffic
    launch_stencil(d_u, d_un, d_k, d_S, p, 2, ny - 1, 2, nx - 1, s_int)
    # Stage C: wait, H2D, unpack, boundary ring
    complete_halos_async(reqs, d_u, bufs, ctx, s_bnd)
    launch_stencil(d_u, d_un, d_k, d_S, p, 1, 1, 1, nx, s_bnd)            # row 1
    launch_stencil(d_u, d_un, d_k, d_S, p, ny, ny, 1, nx, s_bnd)          # row ny
    launch_stencil(d_u, d_un, d_k, d_S, p, 2, ny - 1, 1, 1, s_bnd)        # column 1
    launch_stencil(d_u, d_un, d_k, d_S, p, 2, ny - 1, nx, nx, s_bnd)      # column nx
    s_bnd.synchronize()
    s_int.synchronize()


# ==============================================================================
# TASK 5
# ==============================================================================
def global_reductions(d_old, d_new, ctx, p, with_residual=True):
    ny, nx = ctx["ny"], ctx["nx"]
    cart = ctx["cart_comm"]
    new = d_new.copy_to_host()[1:ny + 1, 1:nx + 1]
    e_loc = np.array([new.sum() * p["dx"] * p["dy"]])
    if with_residual:
        old = d_old.copy_to_host()[1:ny + 1, 1:nx + 1]
        r_loc = np.array([(((new - old) / p["dt"]) ** 2).sum() * p["dx"] * p["dy"]])
    else:
        r_loc = np.array([0.0])
    e_g = np.zeros(1)
    r_g = np.zeros(1)
    cart.Allreduce(e_loc, e_g, op=MPI.SUM)
    cart.Allreduce(r_loc, r_g, op=MPI.SUM)
    return float(e_g[0]), (math.sqrt(r_g[0]) if with_residual else None)


def run_mode(ctx, mode, steps, report=True):
    comm = ctx["comm"]
    h_u, h_k, h_S, dx, dy = initialize_subdomains(ctx)
    kmax_loc = np.array([h_k[1:-1, 1:-1].max()])
    kmax = np.zeros(1)
    ctx["cart_comm"].Allreduce(kmax_loc, kmax, op=MPI.MAX)
    kmax = float(kmax[0])
    dt_max = dx * dx / (4.0 * kmax)
    p = {"dx": dx, "dy": dy, "dt": CFL_SAFETY * dt_max,
         "inv_dx2": 1.0 / (dx * dx), "inv_dy2": 1.0 / (dy * dy),
         "kmax": kmax, "dt_max": dt_max}

    d_u = cuda.to_device(h_u)
    d_un = cuda.to_device(h_u.copy())
    d_k = cuda.to_device(h_k)
    d_S = cuda.to_device(h_S)
    bufs = HaloBuffers(ctx["ny"], ctx["nx"])
    s_int = cuda.stream()
    s_bnd = cuda.stream()
    step_fn = step_overlapped if mode == "overlapped" else step_serialized

    trace = []
    e0, _ = global_reductions(d_u, d_u, ctx, p, with_residual=False)
    trace.append({"step": 0, "energy": e0, "residual_l2": None})

    cuda.synchronize()
    comm.Barrier()
    ev0 = cuda.event(timing=True)
    ev1 = cuda.event(timing=True)
    t0 = time.perf_counter()
    ev0.record(s_bnd)
    for s in range(1, steps + 1):
        step_fn(d_u, d_un, d_k, d_S, bufs, ctx, p, s_int, s_bnd)
        if s % 100 == 0:
            e, r = global_reductions(d_u, d_un, ctx, p)
            trace.append({"step": s, "energy": e, "residual_l2": r})
        d_u, d_un = d_un, d_u
    ev1.record(s_bnd)
    ev1.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    ev_ms = cuda.event_elapsed_time(ev0, ev1)
    wall_ms = comm.allreduce(wall_ms, op=MPI.MAX)
    ev_ms = comm.allreduce(ev_ms, op=MPI.MAX)

    if report and ctx["rank"] == 0:
        print(f"[{mode}] kappa_max={kmax:.6f} dt={p['dt']:.6e} dt_max={dt_max:.6e}")
        for t in trace:
            r = "n/a" if t["residual_l2"] is None else f"{t['residual_l2']:.6e}"
            print(f"[{mode}] step {t['step']:4d}  E={t['energy']:.12e}  L2res={r}")
        print(f"[{mode}] wall={wall_ms:.3f} ms   cuda_event={ev_ms:.3f} ms")

    return {"final": d_u.copy_to_host(), "trace": trace, "wall_ms": wall_ms,
            "event_ms": ev_ms, "p": p}


def ptx_audit():
    counts = {"st.shared.f64": 0, "ld.shared.f64": 0, "bar.sync": 0}
    asm = tiled_stencil_kernel.inspect_asm()
    for ptx in asm.values():
        for key in counts:
            counts[key] += len(re.findall(re.escape(key), ptx))
    passed = all(v > 0 for v in counts.values())
    return counts, passed


def run_solver(github_user):
    ctx = init_distributed_context(github_user)
    rank = ctx["rank"]
    if rank == 0:
        print(f"GITHUB_USERNAME={github_user}")
        print(f"GPU_UUID={ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED={ctx['seed']}")
        print(f"MPI ranks={ctx['size']} (2x2 Cartesian, non-periodic)")
    print(f"rank {rank}: coords={ctx['coords']} up={ctx['up']} down={ctx['down']} "
          f"left={ctx['left']} right={ctx['right']}", flush=True)

    # warm-up (JIT compile, context creation) - not timed
    run_mode(ctx, "serialized", 2, report=False)
    run_mode(ctx, "overlapped", 2, report=False)

    ser = run_mode(ctx, "serialized", STEPS)
    ovl = run_mode(ctx, "overlapped", STEPS)

    ny, nx = ctx["ny"], ctx["nx"]
    diff_loc = np.array([np.max(np.abs(ser["final"][1:ny + 1, 1:nx + 1]
                                       - ovl["final"][1:ny + 1, 1:nx + 1]))])
    diff = np.zeros(1)
    ctx["cart_comm"].Allreduce(diff_loc, diff, op=MPI.MAX)
    parity = float(diff[0])

    if rank == 0:
        counts, passed = ptx_audit()
        print(f"PTX counts: {counts}  ptx_audit_passed={passed}")
        print(f"parity_max_diff={parity:.3e}")
        print(f"serialized_ms={ser['wall_ms']:.3f}  overlapped_ms={ovl['wall_ms']:.3f}")
        manifest = {
            "github_username": github_user,
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["seed"],
            "mpi_ranks": ctx["size"],
            "process_grid": [2, 2],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "steps": STEPS,
            "dt": ser["p"]["dt"],
            "dt_max": ser["p"]["dt_max"],
            "kappa_max": ser["p"]["kmax"],
            "serialized_ms": ser["wall_ms"],
            "overlapped_ms": ovl["wall_ms"],
            "cuda_event_serialized_ms": ser["event_ms"],
            "cuda_event_overlapped_ms": ovl["event_ms"],
            "energy_trace_serialized": ser["trace"],
            "energy_trace_overlapped": ovl["trace"],
            "ptx_counts": counts,
            "ptx_audit_passed": bool(passed),
            "parity_max_diff": parity,
            "parity_ok": parity < 1e-12,
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("Wrote submission_manifest.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: distributed_cuda_solver.py <github_username>")
    run_solver(sys.argv[1])
    
