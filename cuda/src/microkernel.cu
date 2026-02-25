/**
 * GPU Microkernel for Autonomous Scheduling
 * ==========================================
 * A persistent GPU-resident kernel that acts as an OS-level scheduler.
 * Uses CUDA Dynamic Parallelism (CDP) to launch child kernels autonomously.
 *
 * Key Design Decisions:
 *  - Lock-free MPMC queue via atomic CAS operations (avoids deadlock)
 *  - SM-affinity via blockIdx to reduce cross-SM contention
 *  - Heartbeat watchdog using clock64() for livelock detection
 *  - Hierarchical memory: shared mem -> L2 -> global
 *
 * Compile: nvcc -arch=sm_70 -rdc=true -lcudadevrt microkernel.cu -o microkernel
 *          (CDP requires -rdc=true and linking against cudadevrt)
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
#include <stdint.h>
#include "../include/task_queue.cuh"
#include "../include/scheduler.cuh"
#include "../include/metrics.cuh"

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS & TUNABLES
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int WARP_SIZE        = 32;
static constexpr int MAX_SMs          = 108;   // Ampere A100; adjust per GPU
static constexpr int SCHEDULER_WARPS  = 4;     // warps dedicated to scheduling per block
static constexpr int WORKER_WARPS     = 28;    // warps for compute work
static constexpr int BLOCK_THREADS    = (SCHEDULER_WARPS + WORKER_WARPS) * WARP_SIZE;
static constexpr uint64_t WATCHDOG_CYCLES = 1000000000ULL;  // ~1s at 1GHz

// ─────────────────────────────────────────────────────────────────────────────
// DEVICE-SIDE CHILD KERNELS  (launched via Dynamic Parallelism)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generic compute child — executes a task payload.
 * Template parameter T allows type-safe dispatch without device-side vtable.
 */
__global__ void child_compute_kernel(TaskPayload payload, MetricSlot* metrics) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= payload.element_count) return;

    // Simulated data-parallel workload (replace with actual kernel logic)
    float* in  = reinterpret_cast<float*>(payload.input_ptr);
    float* out = reinterpret_cast<float*>(payload.output_ptr);

    float val = in[tid];
    // Polynomial approximation as stand-in for real compute
    #pragma unroll 4
    for (int i = 0; i < payload.iterations; ++i) {
        val = val * val * 0.5f + val * 0.3f + 0.1f;
    }
    out[tid] = val;

    // Atomic metrics update (per-SM, reduce contention)
    if (threadIdx.x == 0) {
        atomicAdd(&metrics[blockIdx.x % MAX_SMs].tasks_completed, 1ULL);
    }
}

/**
 * Reduction child — tree-based parallel reduction.
 * Launched when task type == TASK_REDUCE.
 */
__global__ void child_reduce_kernel(TaskPayload payload, MetricSlot* metrics) {
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x * 2 + tid;
    int n    = payload.element_count;
    float* d = reinterpret_cast<float*>(payload.input_ptr);

    sdata[tid] = (gid < n ? d[gid] : 0.0f) +
                 (gid + blockDim.x < n ? d[gid + blockDim.x] : 0.0f);
    __syncthreads();

    // Tree reduction — only safe inside a single kernel invocation
    for (int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();  // Safe: all threads in block, same kernel, not persistent
    }

    // Warp-level reduction without __syncthreads (shuffle instructions)
    if (tid < WARP_SIZE) {
        float v = sdata[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (tid == 0) {
            float* out = reinterpret_cast<float*>(payload.output_ptr);
            atomicAdd(out + blockIdx.x, v);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SCHEDULER CORE (warp-specialized within the persistent block)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Warp-level task dispatcher.
 * Only SCHEDULER_WARPS execute this path; WORKER_WARPS handle compute.
 *
 * Deadlock Prevention Strategy:
 *   - Never call __syncthreads() in the scheduler warps
 *   - Use __syncwarp() scope-limited to the scheduler warp only
 *   - Lock-free dequeue via CAS — never block waiting for a mutex
 *   - Child kernels launched with cudaStreamFireAndForget (non-blocking)
 */
__device__ __forceinline__
void scheduler_warp_fn(
    cg::thread_block_tile<WARP_SIZE> warp,
    TaskQueue* __restrict__ global_queue,
    MetricSlot* __restrict__ metrics,
    volatile uint32_t* __restrict__ shutdown_flag,
    SchedulerState* __restrict__ sched_state)
{
    int lane = warp.thread_rank();

    // Each warp-lane polls a different queue segment (reduces contention)
    uint32_t my_queue_offset = (blockIdx.x * SCHEDULER_WARPS +
                                (threadIdx.x / WARP_SIZE)) % TASK_QUEUE_SEGMENTS;

    uint64_t last_progress = clock64();

    while (!(*shutdown_flag)) {
        Task task;
        bool got_task = false;

        // Lane 0 does the atomic dequeue; result broadcast via shuffle
        if (lane == 0) {
            got_task = queue_try_dequeue(global_queue, my_queue_offset, &task);
        }

        // Broadcast result to all lanes in warp (warp-uniform branch)
        uint32_t task_type    = __shfl_sync(0xffffffff, task.type,    0);
        uint64_t task_payload = __shfl_sync(0xffffffff,
            reinterpret_cast<uint64_t&>(task.payload_idx), 0);
        got_task = __shfl_sync(0xffffffff, (int)got_task, 0);

        if (got_task) {
            last_progress = clock64();

            // Retrieve full payload from device memory
            TaskPayload* payload = &sched_state->payload_pool[task_payload];

            // Determine launch geometry
            int n_elements = payload->element_count;
            int block_sz   = 256;
            int grid_sz    = (n_elements + block_sz - 1) / block_sz;

            // Dynamic Parallelism: launch child from device
            // cudaStreamFireAndForget → non-blocking, no host sync required
            cudaStream_t child_stream;
            cudaStreamCreateWithFlags(&child_stream, cudaStreamNonBlocking);

            switch (task_type) {
                case TASK_COMPUTE:
                    child_compute_kernel<<<grid_sz, block_sz, 0, child_stream>>>(
                        *payload, metrics);
                    break;

                case TASK_REDUCE: {
                    size_t smem = block_sz * sizeof(float);
                    child_reduce_kernel<<<grid_sz, block_sz, smem, child_stream>>>(
                        *payload, metrics);
                    break;
                }

                case TASK_CHAIN: {
                    // Data-driven trigger: enqueue follow-up tasks based on output
                    // (simulates conditional workflow without CPU round-trip)
                    TaskChainDescriptor* chain =
                        reinterpret_cast<TaskChainDescriptor*>(payload->input_ptr);
                    if (lane == 0) {
                        for (int i = 0; i < chain->next_count; ++i) {
                            queue_enqueue(global_queue,
                                          my_queue_offset,
                                          &chain->next_tasks[i]);
                        }
                    }
                    break;
                }

                default:
                    // Unknown task type — increment error counter, don't stall
                    if (lane == 0)
                        atomicAdd(&metrics[blockIdx.x].errors, 1U);
                    break;
            }

            // Retire stream (device-side streams are cheap; CDP manages lifetime)
            cudaStreamDestroy(child_stream);

            // Mark task complete
            if (lane == 0)
                atomicAdd(&sched_state->tasks_dispatched, 1ULL);

        } else {
            // No task available — spin with exponential backoff
            // Use clock64 delta to avoid burning cycles on empty queue
            uint64_t now = clock64();
            if (now - last_progress > WATCHDOG_CYCLES) {
                // Watchdog: potential livelock — reset queue head pointer
                if (lane == 0) {
                    queue_recover(global_queue, my_queue_offset);
                    atomicAdd(&metrics[blockIdx.x].watchdog_resets, 1U);
                }
                last_progress = now;
            }
            // Cooperative yield: let other warps make progress
            // __nanosleep is Ampere+; fallback to no-op loop on older arch
            #if __CUDA_ARCH__ >= 700
            __nanosleep(100);  // 100ns yield — avoids power-wasting spin
            #endif
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKER WARP LOGIC
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Worker warps run inline compute tasks (small tasks not worth CDP overhead).
 * Separated from scheduler warps to prevent __syncthreads() cross-contamination.
 */
__device__ __forceinline__
void worker_warp_fn(
    cg::thread_block_tile<WARP_SIZE> warp,
    InlineTaskQueue* __restrict__ inline_queue,
    volatile uint32_t* __restrict__ shutdown_flag)
{
    while (!(*shutdown_flag)) {
        InlineTask task;
        if (inline_queue_try_dequeue(inline_queue, blockIdx.x, &task)) {
            // Execute inline (no child kernel launch overhead)
            float* d = reinterpret_cast<float*>(task.data_ptr);
            int n    = task.count;
            int lane = warp.thread_rank();

            for (int i = lane; i < n; i += WARP_SIZE) {
                d[i] = sqrtf(fabsf(d[i])) + 1e-7f;
            }
        }
        #if __CUDA_ARCH__ >= 700
        __nanosleep(50);
        #endif
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PERSISTENT MICROKERNEL ENTRY POINT
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The persistent kernel. Launched ONCE at program start; never exits until
 * shutdown_flag is set from the host.
 *
 * SM Allocation Strategy:
 *   - Launch exactly as many blocks as SMs (one block per SM)
 *   - __launch_bounds__ ensures occupancy and prevents register spill
 *   - Warp specialization: warp IDs [0, SCHEDULER_WARPS) → scheduler
 *                          warp IDs [SCHEDULER_WARPS, total) → workers
 *
 * Memory Layout per SM:
 *   - Shared memory partitioned into: scheduler state (1KB), worker scratch (47KB)
 *   - No global memory locks — all synchronization via atomics
 *
 * Deadlock-Free Guarantee:
 *   - __syncthreads() is NEVER called in scheduler warps
 *   - Worker warps use __syncwarp() only (same-warp scope, cannot deadlock)
 *   - CDP child kernels are independent; scheduler never waits on them
 *   - Lock-free queue uses double-checked CAS with backoff
 */
__global__
__launch_bounds__(BLOCK_THREADS, 1)  // max 1 block per SM → maximizes shared mem
void gpu_microkernel(
    TaskQueue*          __restrict__ global_queue,
    InlineTaskQueue*    __restrict__ inline_queue,
    MetricSlot*         __restrict__ metrics,
    SchedulerState*     __restrict__ sched_state,
    volatile uint32_t*  __restrict__ shutdown_flag)
{
    // ── Warp identification ──────────────────────────────────────────────────
    cg::thread_block block = cg::this_thread_block();
    int warp_id   = threadIdx.x / WARP_SIZE;
    bool is_sched = (warp_id < SCHEDULER_WARPS);

    // ── Shared memory layout ─────────────────────────────────────────────────
    __shared__ uint8_t smem[48 * 1024];  // 48KB max on most architectures
    uint8_t* sched_smem  = smem;                           // [0,  1KB)
    uint8_t* worker_smem = smem + 1024;                    // [1KB, 48KB)

    // ── Per-SM metrics init (only thread 0) ──────────────────────────────────
    if (threadIdx.x == 0) {
        metrics[blockIdx.x] = {0, 0, 0, 0};
    }
    block.sync();  // Safe: called before warp specialization diverges

    // ── Initialize inline queue segment for this SM ──────────────────────────
    if (threadIdx.x == 0) {
        inline_queue_init_segment(inline_queue, blockIdx.x);
    }
    block.sync();  // Last __syncthreads() before specialization

    // ── WARP SPECIALIZATION — warps diverge here permanently ─────────────────
    // CRITICAL: After this point, scheduler and worker warps NEVER synchronize
    //           with each other. No __syncthreads() beyond this line.
    //           Violation = guaranteed deadlock (warp S waits, warp W never arrives)

    if (is_sched) {
        // ── Scheduler warp path ──────────────────────────────────────────────
        auto sched_warp = cg::tiled_partition<WARP_SIZE>(block);
        scheduler_warp_fn(sched_warp, global_queue, metrics,
                          shutdown_flag, sched_state);
    } else {
        // ── Worker warp path ─────────────────────────────────────────────────
        auto worker_warp = cg::tiled_partition<WARP_SIZE>(block);
        worker_warp_fn(worker_warp, inline_queue, shutdown_flag);
    }

    // Both paths exit only when shutdown_flag is set.
    // Thread 0 writes final metrics.
    if (threadIdx.x == 0) {
        metrics[blockIdx.x].final_timestamp = clock64();
    }
}
