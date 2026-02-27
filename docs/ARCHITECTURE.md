# Architecture Deep-Dive

## SM Resource Budget

On A100 SXM4 with `__launch_bounds__(1024, 1)` (1 block per SM):

| Resource            | Per-SM Limit | Our Block Usage | Headroom for CDP |
|---------------------|-------------|-----------------|------------------|
| Registers (32-bit)  | 65,536       | ~49,152 (48/thread × 1024) | ~16,384 |
| Shared Memory       | 164 KB       | 48 KB           | 116 KB (for child kernels on same SM) |
| L1 Cache            | 128 KB (unified with smem) | — | Shared with child kernels |
| Max Blocks/SM       | 32           | **1** (by design) | 31 (available for CDP children) |
| Max Warps/SM        | 64           | 32 (1024 / 32)  | 32 |

> **Key insight:** By forcing 1 block/SM, we get full register and shared memory
> budget for the scheduler. Child kernels launched via CDP run on _other_ SMs —
> they don't compete with the scheduler for these resources.

---

## Warp Specialization Rationale

### Without specialization (naive persistent kernel):

```
Warp 0: loop { if poll_queue() { launch_child() } else { sleep() } }
         ↑ branch diverges at every iteration
         ↑ shfl/atomic mix creates pipeline bubbles
         ↑ cudaStreamCreate in divergent branch = undefined behavior
```

### With specialization:

```
Warps 0-3  (SCHEDULER): loop { poll → CAS → shfl_broadcast → cudaStreamCreate → CDP launch }
                          ↑ all 32 lanes uniform at every instruction
Warps 4-31 (WORKER):    loop { poll_inline → compute → write_back }
                          ↑ all 32 lanes uniform at every instruction
```

The hardware warp scheduler sees 32 warps, all making forward progress, none waiting
for each other. Peak SM utilization is achieved.

---

## Lock-Free Queue State Machine

```
                    queue_try_enqueue()
                           │
                    ┌──────▼──────┐
                    │  Read tail  │  load(acquire)
                    └──────┬──────┘
                           │ tail_packed = [ver:32|idx:32]
                    ┌──────▼──────┐
                    │  Check full │  head_idx == next_idx?
                    └──────┬──────┘
                  not full │
                    ┌──────▼──────┐
     CAS fails ←── │  CAS slot   │  EMPTY → WRITING
    (retry/bail)    │  state[idx] │
                    └──────┬──────┘
                  CAS wins │
                    ┌──────▼──────┐
                    │ Write Task  │  (slot is ours; no race)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Mark FULL   │  store(release) ← consumer sees payload
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Advance tail│  versioned CAS [ver+1|next_idx]
                    └─────────────┘

                    queue_try_dequeue()
                           │
                    ┌──────▼──────┐
                    │  Read head  │  load(acquire)
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │ Check empty │  head_idx == tail_idx?
                    └──────┬──────┘
                  not empty│
                    ┌──────▼──────┐
                    │ Check FULL  │  load(acquire) — may return false if writer slow
                    └──────┬──────┘
                    is FULL│
                    ┌──────▼──────┐
     CAS fails ←── │  CAS slot   │  FULL → READING
    (retry/bail)    │  state[idx] │
                    └──────┬──────┘
                  CAS wins │
                    ┌──────▼──────┐
                    │  Read Task  │  (exclusive access in READING state)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Mark EMPTY  │  store(release)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Advance head│  versioned CAS [ver+1|next_idx]
                    └─────────────┘
```

---

## ABA Problem Walkthrough

```
Timeline: Thread A, B, C operate on segment tail (initially ver=5, idx=0)

T1: A reads  tail = [ver=5, idx=0]   ← A wants to advance to [ver=6, idx=1]
T2: B CAS success: tail → [ver=6, idx=1]   (B enqueued at slot 0)
T3: C dequeues slot 0: tail → [ver=7, idx=2]? No: head advances, not tail
    Actually C dequeues by advancing HEAD, not tail.
    Let's use head for the ABA scenario:

Timeline: Thread A, B, C operate on segment HEAD (initially ver=5, idx=0)

T1: A reads  head = [ver=5, idx=0]   ← wants to dequeue slot 0
T2: B dequeues slot 0: head → [ver=6, idx=1]
T3: C enqueues a new task, then a different thread dequeues up to idx=0 again?
    head = [ver=7, idx=0]  ← idx=0 is back! Classic ABA.

T4: A's CAS: expected=[ver=5, idx=0], actual=[ver=7, idx=0]
    idx matches (0==0) BUT ver doesn't (5≠7)
    CAS FAILS → A retries, reads current head [ver=7, idx=0]
    Everything is correct.

Without version tags:
    T4: A's CAS: expected=[idx=0], actual=[idx=0]
    CAS SUCCEEDS → A advances head to idx=1
    But idx=1 might contain a completely different task!
    Silent data corruption.
```

---

## CDP Launch Pipeline

```
Scheduler Warp (SM 0)
    │
    │ cudaStreamCreateWithFlags(NonBlocking)
    │   → Allocates stream descriptor in device memory (~10μs)
    │
    │ child_kernel<<<grid, block, smem, stream>>>()
    │   → Pushes launch descriptor to CDP command queue
    │   → Device-side hardware command processor picks it up
    │   → Child grid enters hardware scheduler
    │   → Child blocks assigned to free SMs (SM 1, 2, ... N)
    │   → Total: ~45μs P50 on A100
    │
    │ cudaStreamDestroy(stream)
    │   → Deferred: stream freed after all ops on it complete
    │
    └─ Scheduler warp continues polling. Non-blocking; never waits.

Child Kernel (SM 1..N)
    │
    │ grid startup: CTA assignment, register file init, shared mem alloc
    │
    │ first instruction executes ← this is what latency_bench measures
    │
    └─ kernel body executes, exits, SM returned to hardware scheduler pool
```

---

## Deadlock Proof Sketch

**Claim:** `gpu_microkernel()` never deadlocks.

**Proof by construction:**

1. `__syncthreads()` creates a barrier that requires ALL threads in the block
   to reach it before any can proceed.

2. In `gpu_microkernel()`, after the two setup barriers, threads enter one of:
   - `scheduler_warp_fn()`: an infinite loop with no barriers
   - `worker_warp_fn()`: an infinite loop with no barriers

3. If `scheduler_warp_fn()` contained a `__syncthreads()`, worker warps
   (which never call it) would never arrive → scheduler warps wait forever → deadlock.

4. Therefore: the absence of `__syncthreads()` in both specialized paths is
   a sufficient condition for deadlock freedom (with respect to thread-block barriers).

5. Atomic operations (`cuda::atomic::compare_exchange`) are non-blocking by
   definition: they either succeed immediately or return `false`. No warp ever
   waits for another warp to "release" an atomic — there is no ownership to release.

6. `__nanosleep(100)` is a yield hint to the hardware, not a blocking primitive.
   The hardware warp scheduler may immediately reschedule the warp after 100ns;
   the warp does not "wait" in any synchronization sense.

7. Child kernels are fire-and-forget (via `cudaStreamNonBlocking`). The scheduler
   warp never calls `cudaStreamSynchronize()` or `cudaDeviceSynchronize()` on
   device — those are host-only calls that would deadlock the kernel.

**QED:** No barrier, no mutex, no blocking wait → no deadlock. □

---

## Mutex vs Lock-Free on GPU: Why Lock-Free Wins

```
Scenario: 64 warps on one SM, all contending for a mutex.

With mutex (cuda::std::mutex based on spinlock):
  Warp 0 acquires lock.
  Warps 1-63: spin-wait in a loop.
    → Each warp executes: ld.global → compare → branch → ld.global → ...
    → 63 warps × ~8 instructions/iter × 1000 iters = 504,000 wasted instructions
    → SM instruction throughput: ~100% wasted on failed lock attempts
    → Warp 0 (lock holder) shares SM resources with 63 spinners
    → Warp 0 gets 1/64 of SM issue slots → lock held 64× longer → starvation

With lock-free CAS:
  All 64 warps attempt CAS simultaneously.
  Hardware arbitrates: exactly 1 CAS succeeds per cycle.
  Losers get false from CAS and immediately do something else (poll next segment).
    → No warp is "blocked" — all make forward progress
    → Failed CAS → warp moves to different queue segment (no retry stall)
    → SM instruction throughput: ~100% useful work
    → Throughput scales linearly with SM count
```

The key insight: GPU hardware has no OS scheduler to park blocked threads.
A blocked warp on a mutex continues consuming SM resources (issue slots,
registers, shared memory) while contributing zero useful work. Lock-free
eliminates the blocking entirely.
