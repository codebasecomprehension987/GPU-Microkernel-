# GPU Microkernel for Autonomous Scheduling

A GPU-resident persistent kernel acting as an OS-level task scheduler, using
CUDA Dynamic Parallelism (CDP) to launch child kernels autonomously — with zero
CPU intervention after the initial launch.

---

## Architecture Overview

```
HOST (CPU)
  │
  │  cudaMallocManaged (shared queue)
  │  Persistent kernel launch → fires once, never returns
  │  UVM write → inject tasks into shared queue
  │
  └─────────────────────────────────────────────────────────┐
                                                             ▼
                        GPU — Persistent Microkernel
             ┌───────────────────────────────────────────────┐
             │   SM 0          SM 1    ...    SM N            │
             │ ┌─────────┐  ┌─────────┐    ┌─────────┐      │
             │ │Block 0  │  │Block 1  │    │Block N  │      │
             │ │         │  │         │    │         │      │
             │ │Warp 0-3 │  │Warp 0-3 │    │Warp 0-3 │      │
             │ │SCHEDULER│  │SCHEDULER│    │SCHEDULER│      │
             │ │         │  │         │    │         │      │
             │ │Warp 4-31│  │Warp 4-31│    │Warp 4-31│      │
             │ │ WORKERS │  │ WORKERS │    │ WORKERS │      │
             │ └────┬────┘  └────┬────┘    └────┬────┘      │
             │      │            │               │            │
             │   Dynamic Parallelism (CDP) child kernel launches
             │      │            │               │            │
             │ ┌────▼────┐  ┌───▼─────┐    ┌───▼─────┐     │
             │ │child_   │  │child_   │    │child_   │     │
             │ │compute  │  │reduce   │    │compute  │     │
             │ └─────────┘  └─────────┘    └─────────┘     │
             └───────────────────────────────────────────────┘
                              Lock-Free MPMC Queue
                         ┌──────────────────────────┐
                         │  Segment 0   Segment 1   │
                         │  [head/tail] [head/tail] │  ← Atomic CAS
                         │  [slot_state][slot_state] │  ← EMPTY/WRITING/FULL/READING
                         │  [payloads]  [payloads]  │
                         └──────────────────────────┘
```

---

## Why This Is Extremely Hard

### 1. Resource Contention at the SM Level

A GPU SM has fixed resources shared by all resident warps:

| Resource            | Per-SM Limit (A100) | Our Usage             |
|---------------------|---------------------|-----------------------|
| Registers           | 65,536              | ~128 per thread × 32 warps = 65,536 (at limit) |
| Shared Memory       | 164 KB              | 48 KB per block       |
| L1 Cache            | 164 KB (unified)    | Contended by persistent + child kernels |
| Max Resident Blocks | 32                  | 1 (we use all resources for one block) |
| Max Resident Warps  | 64                  | 32 (our BLOCK_THREADS = 1024) |

The scheduler block occupies the entire SM. Child kernels launched via CDP
must run on **other SMs** — creating cross-SM traffic and resource pressure.
If all SMs are occupied by the persistent kernel, CDP child kernels queue
until a slot opens.

**Solution:** Use `__launch_bounds__(BLOCK_THREADS, 1)` to tell the compiler
to assume max 1 block/SM, enabling it to maximize register/smem usage without
the usual multi-block sharing constraints.

### 2. Deadlock Prevention

The classic CUDA `__syncthreads()` requires **all threads in the block** to
reach the barrier. In a persistent kernel with warp specialization, scheduler
warps and worker warps take different infinite loops — a `__syncthreads()` in
either path would cause the block to hang forever.

**Our solution — The Three Rules:**

```
Rule 1: __syncthreads() is called EXACTLY TWICE in the entire persistent kernel:
        once before warp specialization, never after.

Rule 2: Scheduler warps use ONLY:
        - __shfl_sync()   (warp-scope, no cross-warp barrier)
        - __nanosleep()   (yield, not barrier)
        - Atomic ops      (lock-free, no blocking)

Rule 3: Worker warps use ONLY:
        - __syncwarp()    (warp-scope only, 32 threads)
        - Warp shuffles   (same warp, no barrier)
```

CDP child kernels are independent execution contexts — they have their own
`__syncthreads()` barriers which are perfectly safe (all threads in the child
kernel reach them).

### 3. ABA Problem in the Lock-Free Queue

```
Thread A reads tail = (ver=5, idx=10)
Thread B dequeues slot 10 (ver→6)
Thread C enqueues slot 10 (ver→7)  ← ABA: slot 10 reused!
Thread A's CAS succeeds: tail (ver=5,idx=10) → (ver=6,idx=11)
                         ^^^ WRONG: version 5 is stale
```

**Solution:** 64-bit fat pointer packs `[version:32 | index:32]`. Thread A's
CAS fails because the version changed (5 ≠ 7), even though the index is the
same. The version counter is monotonically increasing — it never wraps in
practice (2³² = 4 billion cycles per slot).

### 4. Rust Safety on the Device

`rust-cuda` brings Rust's ownership model to GPU kernels, catching a class
of bugs at compile time:

| Bug Type                    | C++ Detection | Rust-CUDA Detection |
|-----------------------------|---------------|---------------------|
| Host ptr used on device     | Runtime crash | **Compile error**   |
| Missing `__device__` attr   | Linker error  | **Compile error**   |
| Using `Box`/`Vec` on device | Undefined     | **Compile error**   |
| Data race on shared memory  | Undefined     | Partial (unsafe)    |
| Integer overflow             | Undefined     | **Debug panic**     |

**What Rust cannot save you from:**
- Warp divergence (performance issue, not safety)
- Cross-warp data races through raw pointers (still `unsafe`)
- CUDA memory ordering violations (must use `cuda::atomic` scopes manually)
- Deadlocks from misuse of `sync_threads()` (same rules apply)

---

## Complexity Analysis

### Task Queue Operations

| Operation     | Best Case | Worst Case | Notes |
|---------------|-----------|------------|-------|
| `enqueue()`   | O(1)      | O(k)       | k = retries on CAS contention |
| `dequeue()`   | O(1)      | O(k)       | k = wait for SLOT_WRITING→FULL |
| `recover()`   | O(N)      | O(N)       | N = queue capacity (1024) |

### Scheduler Throughput

Per SM per second (theoretical, A100 @ 1.41 GHz):

```
Scheduler warps: 4
Instructions per poll: ~20 (load, CAS, branch)
Warp issue rate: 1 instruction / 4 cycles (compute-bound)

Poll rate = 4 warps × (1.41 GHz / 20 instr) ≈ 280M polls/sec/SM
With 108 SMs: ≈ 30 Billion polls/sec total

Task dispatch rate limited by:
- CDP launch overhead: ~50μs per child kernel launch (measured)
- Stream creation: ~10μs per stream (amortized with stream pools)
- Peak practical throughput: ~50K child kernel launches/sec
```

### Memory Hierarchy Access Latency

| Memory Type     | Latency    | Bandwidth      | Used For |
|-----------------|------------|----------------|----------|
| Registers       | 0 cycles   | N/A            | Loop vars, warp state |
| Shared Memory   | 1-4 cycles | ~19 TB/s (A100)| SM-local scheduler state |
| L1 Cache        | ~20 cycles | ~19 TB/s       | Hot queue segments |
| L2 Cache        | ~200 cycles| 4 TB/s         | Task payloads |
| HBM2e (Global) | ~800 cycles| 2 TB/s         | Task queue, metrics |
| UVM (PCIe)     | ~10μs      | 32 GB/s        | Host→GPU task injection |

### Child Kernel Parallelism

At steady state with N SMs:
- 1 SM hosts the scheduler's persistent block
- N-1 SMs available for child kernels
- Child kernel concurrency: up to `(N-1) × max_blocks_per_SM` blocks
- For A100: 107 SMs × 32 blocks × 1024 threads ≈ 3.5M concurrent child threads

---

## Build Instructions

### Prerequisites

```bash
# CUDA Toolkit 11.8+ (for cuda::atomic, __nanosleep, CDP v2)
# Rust nightly (for PTX backend)
# cmake 3.20+

# Verify CDP support
nvidia-smi --query-gpu=compute_cap --format=csv
# Must be >= 3.5
```

### CUDA C++ Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./microkernel
```

### Rust Build (rust-cuda)

```bash
# Install the rust-cuda toolchain
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
cargo install cuda-builder

# Build PTX (device code)
cargo build --release --target nvptx64-nvidia-cuda

# Build host binary
cargo build --release
./target/release/host
```

### Analysis Targets

```bash
# Profiling with Nsight Compute
make profile

# Memory error checking
make memcheck

# Race condition detection
make racecheck

# Inspect generated PTX assembly
make ptx
cat build/microkernel.ptx | grep -A5 "queue_try_dequeue"
```

---

## Key Design Decisions

### Why Persistent Kernel vs. Normal Kernel?

| Approach          | Latency to Start New Work | CPU Involvement | Overhead |
|-------------------|--------------------------|-----------------|----------|
| Normal kernel     | ~10μs (launch overhead)  | Every task      | High     |
| Persistent kernel | ~1μs (queue poll)        | Never (after init) | Low   |
| Persistent + CDP  | ~50μs (child launch)     | Never           | Medium   |

For latency-sensitive workloads (trading, robotics, real-time vision), the
persistent kernel eliminates the CPU round-trip that normal scheduling requires.

### Why Lock-Free Over Mutex?

CUDA does have `cuda::std::mutex` (CUDA 11.4+), but on the device:
- A mutex spin-locks a warp while other warps in the same SM continue
- If the lock-holder's warp is descheduled, all waiters burn cycles
- With 32 warps/SM and 4 scheduler warps holding locks, throughput collapses

Lock-free CAS allows the hardware warp scheduler to continue making progress
on other warps while a CAS retry is in flight. The hardware's ability to
switch warps every cycle is what makes lock-free preferable to mutex.

### Why Warp Specialization?

Alternative: run scheduler and worker logic in the same warp, interleaved.
Problem: the scheduler loop's branch divergence (got_task? yes: launch, no: spin)
would cause warp-level serialization on every iteration.

Warp specialization guarantees:
- Scheduler warps: 100% branch-coherent (uniform control flow in warp)
- Worker warps: 100% branch-coherent (same computation on all lanes)
- No divergence tax anywhere in the hot path

---

## Known Limitations & Future Work

1. **CDP Launch Latency:** ~50μs per child kernel is high for micro-tasks.
   Mitigation: use the inline task queue for tasks with element_count < 256.

2. **SM Starvation:** If all SMs run child kernels, the persistent scheduler
   blocks cannot be rescheduled. Mitigation: reserve 1 SM for scheduler via
   CUDA MPS (Multi-Process Service) partitioning.

3. **Rust ABI Stability:** The `nvptx64-nvidia-cuda` target in rust-cuda is
   experimental. The FFI boundary between Rust PTX and CUDA C++ requires
   careful `#[repr(C)]` annotation and manual ABI matching.

4. **Hopper (sm_90) Changes:** The `__nanosleep()` semantics changed slightly
   on H100 (Thread Block Clusters). The scheduler may need updating for
   cluster-aware scheduling.

5. **Dynamic Parallelism v2 (CDP2):** CDP2 (CUDA 11.3+) is used here for
   better stream semantics. CDP1 (sm_35-sm_70) has different stream restrictions.
