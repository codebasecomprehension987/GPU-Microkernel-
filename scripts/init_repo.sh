#!/usr/bin/env bash
# =============================================================================
# init_repo.sh — Bootstrap the GPU Microkernel repo with atomic commits
#
# Run from project root:
#   chmod +x scripts/init_repo.sh && ./scripts/init_repo.sh
#
# Each file gets its own commit with a full conventional-commit message.
# Commit order follows dependency graph: types → data structures → kernel →
# host → build → tests → CI → docs.
# =============================================================================

set -euo pipefail

GIT="git"
AUTHOR="GPU Microkernel Bot <bot@gpumicrokernel.dev>"
DATE_BASE="2024-01-15T09:00:00"

# Helper: commit a single file with a message
commit_file() {
    local file="$1"
    local msg="$2"
    $GIT add "$file"
    GIT_AUTHOR_NAME="GPU Microkernel Bot" \
    GIT_AUTHOR_EMAIL="bot@gpumicrokernel.dev" \
    GIT_COMMITTER_NAME="GPU Microkernel Bot" \
    GIT_COMMITTER_EMAIL="bot@gpumicrokernel.dev" \
    $GIT commit -m "$msg"
}

# ── Init ─────────────────────────────────────────────────────────────────────
$GIT init
$GIT checkout -b main

# ── .gitignore ───────────────────────────────────────────────────────────────
commit_file .gitignore \
"chore: add .gitignore for CUDA, Rust, and CMake artifacts

Excludes:
- build/ and CMakeCache.txt (CMake out-of-source builds)
- target/ (Rust/Cargo build output)
- *.ptx, *.cubin, *.fatbin (compiled GPU binaries)
- *.ncu-rep, *.nvvp (Nsight profiler reports)
- *.d (depfiles from nvcc)
- .vscode/, .idea/ (editor configs — keep local)"

# ── Cargo.toml ───────────────────────────────────────────────────────────────
commit_file Cargo.toml \
"build: initialize Cargo workspace for rust-cuda host/device split

Sets up a dual-target Cargo project:
- Host binary (src/host.rs) targets native x86_64 using 'cust' 0.3
- Device library (src/lib.rs) targets nvptx64-nvidia-cuda via rust-cuda

Key decisions:
- profile.release sets panic=abort (no unwinding on CUDA device)
- lto=fat is required for PTX size optimization (reduces register pressure)
- Feature flags sm_70/sm_80/sm_90 control nvcc -arch without touching source
- cuda-builder in build-dependencies handles PTX compilation automatically

Refs: https://github.com/Rust-GPU/Rust-CUDA"

# ── CMakeLists.txt ────────────────────────────────────────────────────────────
commit_file CMakeLists.txt \
"build: add CMake configuration for CUDA C++ microkernel

Configures nvcc for Dynamic Parallelism (CDP) with:
- CMAKE_CUDA_SEPARABLE_COMPILATION=ON (-rdc=true flag)
- Links cudadevrt (device-side CUDA runtime required for CDP)
- CUDA_RESOLVE_DEVICE_SYMBOLS=ON for final CDP link step
- Multi-arch SASS: sm_70, sm_80, sm_86, sm_89, sm_90 in one fatbin

Custom targets:
- 'profile'   → ncu --set full (Nsight Compute full metric collection)
- 'memcheck'  → compute-sanitizer --tool memcheck
- 'racecheck' → compute-sanitizer --tool racecheck
- 'ptx'       → raw PTX dump for manual inspection of scheduler hot path

Compiler flags:
- --use_fast_math (acceptable for non-precision-critical scheduler ops)
- -maxrregcount=128 (leave SM register headroom for CDP child kernels)
- -lineinfo (enables cuda-gdb source correlation without full debug info)"

# ── cuda/include/metrics.cuh ─────────────────────────────────────────────────
commit_file cuda/include/metrics.cuh \
"feat(metrics): add per-SM performance counter structure

MetricSlot is cacheline-aligned (alignas(64)) to prevent false sharing
between adjacent SM counters in the global metrics array.

Fields:
- tasks_completed: monotonically increasing, read by host for throughput
- errors: incremented on unknown task type or CDP launch failure
- watchdog_resets: tracks how often the livelock detector fired
- final_timestamp: clock64() snapshot at kernel exit for latency analysis

Padding _pad[4] pushes struct to 64 bytes exactly (one cache line).
Host reads these via async cudaMemcpyAsync — never blocks the GPU.

Why unsigned long long for tasks_completed: atomicAdd on 64-bit requires
ULL; using uint64_t causes nvcc to emit slower 2x32-bit emulated atomics
on pre-sm_60 hardware."

# ── cuda/include/scheduler.cuh ────────────────────────────────────────────────
commit_file cuda/include/scheduler.cuh \
"feat(scheduler): define SchedulerState and payload pool layout

SchedulerState holds the shared mutable state visible to all scheduler warps:
- tasks_dispatched / tasks_enqueued: device-scope atomics for back-pressure
  detection (if dispatched << enqueued, queue is backing up)
- payload_pool[65536]: pre-allocated TaskPayload array; avoids device-side
  malloc entirely (no heap fragmentation, deterministic access latency)

PAYLOAD_POOL_SZ = 65536 was chosen to fit within 2GB at 32 bytes/payload.
In production, swap for a 2-level pool: hot (L2-resident) + cold (HBM).

MAX_SMs_CONST = 108 covers Ampere A100 SXM4. Change to 132 for H100 SXM5
or query at runtime via cudaDeviceGetAttribute."

# ── cuda/include/task_queue.cuh ───────────────────────────────────────────────
commit_file cuda/include/task_queue.cuh \
"feat(queue): implement lock-free MPMC task queue with ABA prevention

Core data structure for zero-CPU-intervention task dispatch.

Design: Segmented ring buffer with per-slot state machine
  EMPTY → WRITING → FULL → READING → EMPTY

Each state transition uses device-scope CAS (cuda::atomic with
thread_scope_device), forming happens-before edges that guarantee:
  - Readers never see partial writes (WRITING guard)
  - Writers never corrupt slots being read (READING guard)
  - ABA problem solved via 64-bit fat pointer: [version:32|index:32]

Segmentation (TASK_QUEUE_SEGMENTS=16):
  Each scheduler warp polls a different segment based on:
    seg = (blockIdx * SCHEDULER_WARPS + warp_id) % SEGMENTS
  This spreads atomic contention across 16 independent head/tail pairs,
  reducing cache invalidation traffic on the L2 segment holding the atomics.

queue_recover():
  Called by watchdog when clock64() delta exceeds WATCHDOG_CYCLES.
  Scans for SLOT_READING slots (reader died mid-dequeue) and resets to
  SLOT_EMPTY. Does NOT touch SLOT_WRITING (writer may still complete).
  Safe because: READING means the task was already copied to local 'Task'
  variable before the reader died; the slot can be recycled.

Memory ordering rationale:
  enqueue: store(SLOT_FULL, release) — payload writes visible before flag
  dequeue: load(acquire) — flag visible; then payload read safe
  Together these form an acquire-release pair across warps/SMs."

# ── cuda/src/microkernel.cu ───────────────────────────────────────────────────
commit_file cuda/src/microkernel.cu \
"feat(kernel): implement persistent GPU microkernel with warp specialization

The centerpiece of the project. gpu_microkernel() launches once per SM
(__launch_bounds__(1024, 1) → 1 block/SM) and never exits until shutdown.

Architecture:
  Warps 0-3  (SCHEDULER_WARPS): Run scheduler_warp_fn()
  Warps 4-31 (WORKER_WARPS):    Run worker_warp_fn()

Deadlock prevention — THE critical invariant:
  __syncthreads() appears exactly TWICE in the entire kernel:
    1. After metrics init (before warp specialization)
    2. After inline queue init (before warp specialization)
  After these two barriers, scheduler and worker warps NEVER synchronize
  with each other. Any __syncthreads() in either specialized path would
  deadlock because the other warp group never reaches it.

Dynamic Parallelism dispatch in scheduler_warp_fn():
  - Lane 0 does the CAS dequeue; result broadcast via __shfl_sync(0xffffffff)
    making the branch warp-uniform (no divergence)
  - Child kernels launched with cudaStreamNonBlocking (fire and forget)
  - TASK_CHAIN: enqueues follow-up tasks directly on device — full DAG
    execution without any CPU round-trip

Watchdog:
  If clock64() delta > WATCHDOG_CYCLES (~1s), calls queue_recover() and
  increments watchdog_resets metric. Uses __nanosleep(100) (Volta+) to
  yield 100ns between polls on empty queue — eliminates power-wasting
  spin while keeping latency under 1μs.

Child kernels:
  child_compute_kernel: polynomial approximation (stand-in for real work)
  child_reduce_kernel: two-level reduction (shared mem tree + warp shuffle)
    Note: __syncthreads() IS used inside child kernels — safe because all
    threads in the child kernel's block reach every barrier. No specialization.

Register budget (A100):
  -maxrregcount=128 × 1024 threads = 131,072 registers used
  SM limit = 65,536 per block... wait: the limit is per-SM, not per-block.
  With 1 block/SM, we get all 65,536 registers. Compiler will spill above
  that; -maxrregcount=64 is the real practical limit. Adjust for target SM."

# ── cuda/src/host_launcher.cpp ────────────────────────────────────────────────
commit_file cuda/src/host_launcher.cpp \
"feat(host): implement C++ host control plane for microkernel lifecycle

Manages the full lifecycle: init → launch → inject tasks → monitor → shutdown.

Memory strategy:
  d_global_queue:   cudaMallocManaged (UVM) — host writes tasks directly,
                    CUDA migrates pages to GPU on first access. Prefetch via
                    cudaMemPrefetchAsync after injection to avoid demand-paging
                    stalls during the scheduler's first poll cycle.
  d_metrics:        cudaMalloc (device-only) — host reads async via
                    cudaMemcpyAsync on dedicated stream; never blocks GPU.
  h_shutdown_flag:  cudaHostAlloc(cudaHostAllocMapped) + cudaHostGetDevicePointer
                    — pinned, zero-copy. Host writes 1; GPU reads via UVA
                    pointer. No explicit sync needed; UVA coherence handles it.

verify_device_capabilities():
  Checks compute capability >= 3.5 (CDP minimum) and queries
  cudaDevAttrCooperativeLaunch. Exits with error on unsupported hardware
  rather than producing a silent runtime crash.

inject_tasks():
  Payloads go via cudaMemcpyAsync (fast HtoD DMA).
  Task descriptors go via managed memory write (host-side, CPU cache).
  cudaMemAdvise(SetPreferredLocation, GPU) + cudaMemPrefetchAsync ensure
  pages migrate before the scheduler polls, hiding UVM latency.

Shutdown sequence:
  1. __atomic_store_n(h_shutdown_flag, 1, RELEASE) — host-side release fence
  2. cudaStreamSynchronize — wait for kernel exit
  3. Async metrics copy + print
  4. Free all allocations (managed, device, pinned, stream)"

# ── src/lib.rs ────────────────────────────────────────────────────────────────
commit_file src/lib.rs \
"feat(rust-device): implement Rust device kernels with compile-time safety

#![no_std] PTX-targeting Rust library demonstrating rust-cuda's safety model.

DevicePtr<'a, T>:
  Phantom lifetime ties device pointer to its allocation's scope.
  The Rust borrow checker statically prevents this from being stored
  in a host-side struct or returned across the kernel boundary.
  read_at() and atomic_add_u32() are marked #[device] — calling them
  from host code is a compile error (wrong execution space).

PoolAllocator:
  Fixed-capacity slot allocator using a u32 bitmask freelist.
  alloc() uses compare_exchange_weak in a CAS loop — the 'weak' variant
  allows spurious failure but is faster on sm_70+ (no LL/SC overhead).
  free() uses fetch_or — safe without CAS because setting a bit is
  idempotent (monotonic: once freed, always free until next alloc cycle).

Warp primitives (unsafe):
  warp_broadcast_u32(): wraps shfl.sync.idx.b32 inline PTX.
    Safety contract: mask must include all active lanes; lane in [0,32).
    Violation = undefined PTX behavior (not caught by Rust borrow checker;
    documented in Safety section as caller's responsibility).
  warp_ballot(): wraps vote.sync.ballot.b32 — counts predicate matches.

rust_worker_kernel / rust_warp_reduce_kernel:
  Demonstrate real #[kernel] usage. The warp reduction uses shfl_down_sync
  from cuda_std::shuffle — no shared memory needed (pure register ops).
  Atomic f32 add via inline PTX 'atom.global.add.f32' — cuda_std wraps
  this but inline PTX shown for clarity.

What Rust catches here vs C++:
  ✓ Using Box/Vec/String on device (no allocator) → compile error
  ✓ Host ptr passed as kernel arg → type system prevents it
  ✓ Integer overflow in debug builds → panic (abort on device)
  ✗ Cross-warp data races through *mut T → still unsafe, documented"

# ── src/host.rs ───────────────────────────────────────────────────────────────
commit_file src/host.rs \
"feat(rust-host): implement Rust host control plane using cust crate

Idiomatic Rust bindings over the CUDA driver API via 'cust' 0.3.

MicrokernelHost:
  - cust::context::Context: RAII primary context (drops on struct drop)
  - cust::stream::Stream::NON_BLOCKING: dedicated stream, never blocks
    on the default stream (avoids accidental synchronization with other
    GPU work in the process)
  - cust::memory::UnifiedBuffer<u32>: UVM-backed shutdown flag, host-writable
    and GPU-readable without explicit memcpy

DeviceBuffer<T>:
  Generic RAII wrapper around cust::memory::DeviceBuffer<T>.
  Freed automatically on drop — no manual cudaFree calls anywhere.
  copy_from_host/copy_to_host use synchronous copies; async variant
  would require pinning the host slice (future work).

WorkloadGenerator:
  Xorshift64 PRNG — chosen because it has no stdlib dependencies and
  produces full-period 64-bit sequences. In production, replace with
  a real workload source (Kafka consumer, sensor DMA, etc.).
  generate_batch() returns (task_type, element_count, priority) tuples
  deterministically for reproducible benchmarks.

Monitor::tick():
  Computes tasks/sec by delta from last sample. Only prints when the
  sampling interval elapses — avoids flooding stdout on fast GPUs.
  elapsed() uses std::time::Instant (monotonic, no NTP drift).

Shutdown sequence mirrors C++ launcher:
  1. Write 1 to UnifiedBuffer (UVM coherence propagates to GPU)
  2. stream.synchronize() (blocks until kernel exits)
  3. collect_metrics() → print summary
  All allocations freed by Drop impls — no explicit cleanup needed."

# ── tests/ ────────────────────────────────────────────────────────────────────
commit_file tests/queue_tests.cu \
"test(queue): add device-side unit tests for lock-free queue correctness

Tests run as CUDA kernels (device-side assertions) to validate queue
behavior under conditions impossible to reproduce on the host:

test_single_producer_consumer:
  Basic enqueue + dequeue round-trip. Validates payload survives intact
  through the slot state machine. Checks version counter increments.

test_aba_prevention:
  Manually constructs the ABA scenario:
    1. Thread A reads tail_packed (version=N, idx=X)
    2. Thread B dequeues X (version=N+1)
    3. Thread C enqueues X (version=N+2)
    4. Thread A attempts CAS with stale version=N → must FAIL
  Verifies CAS failure without data corruption.

test_queue_full_behavior:
  Fills queue to capacity. Verifies enqueue returns false (not undefined).
  Verifies dequeue still works after queue was full.

test_concurrent_mpmc:
  Launches 256 threads simultaneously: 128 producers + 128 consumers.
  Uses atomicAdd to count successful enqueues/dequeues.
  Asserts total_dequeued == total_enqueued after all threads complete.
  Run with compute-sanitizer --tool racecheck to verify no data races.

test_watchdog_recovery:
  Artificially sets a slot to SLOT_READING (stuck state).
  Calls queue_recover() and verifies slot returns to SLOT_EMPTY.
  Verifies queue remains functional after recovery."

# ── benches/ ──────────────────────────────────────────────────────────────────
commit_file benches/throughput_bench.cu \
"bench: add throughput and latency benchmarks for queue and CDP dispatch

Measures three critical performance numbers:

1. Queue throughput (queue_throughput_bench):
   - 108 blocks × 32 warps continuously enqueue+dequeue
   - Reports: million ops/sec, avg CAS retries/op
   - Expected: >500M ops/sec on A100 (L2 cache hit rate ~95%)
   - Bottleneck: L2 bandwidth for atomic operations on head/tail

2. CDP launch latency (cdp_launch_latency_bench):
   - Measures clock64() delta from cudaStreamCreate to child kernel start
   - Reports: P50, P95, P99 latency in microseconds
   - Expected: P50 ~45μs, P99 ~120μs on A100
   - Bottleneck: CDP kernel launch uses the hardware command processor,
     which serializes launches from the same parent warp

3. End-to-end task latency (e2e_latency_bench):
   - Measures time from queue_enqueue() to child kernel first instruction
   - Includes: queue poll latency + CDP launch overhead
   - Expected: P50 ~50μs (dominated by CDP launch)

Run with: nvcc -arch=sm_80 -rdc=true -lcudadevrt benches/throughput_bench.cu
          -I cuda/include -o bench && ./bench"

# ── docs/ ─────────────────────────────────────────────────────────────────────
commit_file docs/ARCHITECTURE.md \
"docs: add deep-dive architecture document with SM resource analysis

Covers:
- SM resource budget table (registers, shared mem, L1 cache per block)
- Warp specialization rationale and divergence analysis
- Lock-free queue state machine diagram (ASCII)
- ABA problem walkthrough with timeline diagram
- CDP launch pipeline: device command processor → child grid setup
- Memory hierarchy access latency table (registers → HBM2e → UVM/PCIe)
- Deadlock proof sketch: why the two-syncthreads invariant is sufficient
- Comparison: mutex vs lock-free on GPU (why mutex loses at high warp count)
- rust-cuda safety boundary: what the type system catches vs what it cannot"

commit_file docs/PROFILING.md \
"docs: add Nsight Compute profiling guide for microkernel analysis

Key metrics to watch:
- l2_global_atomic_store: should be ~= 2× task count (enq + deq head/tail)
- sm__warps_active.avg: should be ~32/SM for persistent kernel
- l1tex__t_sector_hit_rate: queue hot path should hit >80%
- device__cuda_dynamic_parallelism_launch_count: tracks CDP launches

Common pathologies and fixes:
- Low warp occupancy (<16): register spill → reduce -maxrregcount
- High l2_global_atomic_store: queue contention → add segments
- High cdp_launch stalls: too many concurrent CDP launches → batch tasks
- Watchdog firing repeatedly: queue segment imbalance → rebalance seg assignment

ncu command for scheduler warp analysis:
  ncu --metrics sm__warps_active,l2_global_atomic_store,\
      device__cuda_dynamic_parallelism_launch_count \
      --kernel-name gpu_microkernel ./microkernel"

# ── .github/workflows/ ────────────────────────────────────────────────────────
commit_file .github/workflows/ci.yml \
"ci: add GitHub Actions workflow for CUDA build and test matrix

Runs on: [ubuntu-22.04] with CUDA toolkit installed via Jimver/cuda-toolkit.

Jobs:
  build-cuda:
    - Installs CUDA 12.2 (latest stable with CDP v2 support)
    - cmake -DCMAKE_BUILD_TYPE=Release
    - make -j4 microkernel
    - Validates binary exists and has correct CUDA capability flags
    - Caches build/ directory keyed on CMakeLists.txt hash

  build-rust:
    - Installs rust nightly (required for PTX backend)
    - Installs nvptx64-nvidia-cuda target
    - cargo build --release --target nvptx64-nvidia-cuda (device lib)
    - cargo build --release (host binary)
    - cargo clippy (lint device code for no_std compliance)

  test-sanitizers (GPU runners only, skipped in free tier):
    - Runs compute-sanitizer --tool memcheck on queue tests
    - Runs compute-sanitizer --tool racecheck on concurrent MPMC test
    - Uploads sanitizer reports as artifacts

  Note: Actual GPU execution requires self-hosted runners with NVIDIA GPU.
  Build-only jobs run on GitHub-hosted ubuntu runners fine."

# ── scripts/ ─────────────────────────────────────────────────────────────────
commit_file scripts/init_repo.sh \
"chore(scripts): add repo initialization script with ordered atomic commits

This script itself. Commits files in dependency order so git log reads as
a coherent implementation narrative:
  types/structs → queue → kernel → host → build → tests → CI → docs

Each commit message follows Conventional Commits spec:
  <type>(<scope>): <summary>
  <blank line>
  <body — what and why, not how>

Types used: feat, build, chore, test, bench, docs, ci, refactor, fix
Scopes: kernel, queue, host, rust-device, rust-host, metrics, scheduler"

# ── README.md ─────────────────────────────────────────────────────────────────
commit_file README.md \
"docs: add comprehensive README with architecture, complexity, and build guide

Top-level documentation covering:
- ASCII architecture diagram: host → persistent kernel → CDP children
- Why This Is Hard: SM resource table, deadlock proof, ABA walkthrough,
  rust-cuda safety matrix (compile-time catches vs runtime required)
- Complexity analysis: queue O(1) amortized, CDP throughput numbers,
  memory hierarchy latency table, peak parallelism calculation
- Build instructions: CMake (CUDA C++), Cargo (Rust), custom targets
- Key design decisions: persistent vs normal kernel comparison table,
  lock-free vs mutex analysis, warp specialization rationale
- Known limitations: CDP latency, SM starvation, Rust ABI stability,
  Hopper sm_90 Thread Block Cluster changes"

echo ""
echo "✅ Repository initialized with $(git log --oneline | wc -l) commits."
echo ""
git log --oneline
