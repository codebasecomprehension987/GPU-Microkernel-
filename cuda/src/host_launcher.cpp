/**
 * host_launcher.cpp — Host Control Plane
 * ========================================
 * Responsible for:
 *   1. Allocating device memory for all microkernel data structures
 *   2. Launching the persistent kernel (once, at startup)
 *   3. Feeding tasks into the queue from the CPU side
 *   4. Signaling shutdown and collecting metrics
 *
 * After launch, the GPU runs autonomously. The host merely:
 *   - Writes new tasks into pinned memory → GPU picks them up via mapped ptr
 *   - Reads metrics via async memcpy (never stalls the GPU)
 *   - Sets shutdown_flag when done
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <chrono>
#include <thread>

// Forward declarations matching CUDA kernel signatures
struct TaskQueue;
struct InlineTaskQueue;
struct MetricSlot;
struct SchedulerState;
struct Task;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA ERROR CHECKING
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERR] %s:%d — %s: %s\n",                 \
                    __FILE__, __LINE__, #call, cudaGetErrorString(_e));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// DEVICE CAPABILITY CHECK
// ─────────────────────────────────────────────────────────────────────────────

void verify_device_capabilities() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Shared mem/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);

    // CDP requires compute capability >= 3.5
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        fprintf(stderr, "ERROR: Dynamic Parallelism requires sm_35 or higher.\n");
        exit(EXIT_FAILURE);
    }

    // Check CDP is actually enabled
    int supports_cdp = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supports_cdp,
        cudaDevAttrCooperativeLaunch, device));
    printf("  Cooperative launch: %s\n", supports_cdp ? "YES" : "NO");

    printf("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// MICROKERNEL CONTEXT
// ─────────────────────────────────────────────────────────────────────────────

struct MicrokernelContext {
    // Device pointers
    TaskQueue*       d_global_queue;
    InlineTaskQueue* d_inline_queue;
    MetricSlot*      d_metrics;
    SchedulerState*  d_sched_state;
    uint32_t*        d_shutdown_flag;

    // Host-mapped shutdown flag (pinned memory for zero-copy access)
    uint32_t*        h_shutdown_flag;

    // Configuration
    int num_sms;
    int block_threads;

    // Kernel stream
    cudaStream_t kernel_stream;
};

// ─────────────────────────────────────────────────────────────────────────────
// ALLOCATION
// ─────────────────────────────────────────────────────────────────────────────

// Sizes defined here to avoid including .cuh in this .cpp
static constexpr size_t TASK_QUEUE_SIZE    = 108 * 1024 * sizeof(uint64_t) * 4; // approx
static constexpr size_t INLINE_QUEUE_SIZE  = 108 * 256  * 16;
static constexpr size_t METRIC_SLOT_SIZE   = 64;  // alignas(64)
static constexpr size_t PAYLOAD_POOL_SLOTS = 65536;
static constexpr size_t PAYLOAD_SIZE       = 32;

MicrokernelContext* microkernel_init() {
    verify_device_capabilities();

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    MicrokernelContext* ctx = new MicrokernelContext();
    ctx->num_sms       = prop.multiProcessorCount;
    ctx->block_threads = 1024;  // (4 + 28) * 32

    // ── Device allocations ────────────────────────────────────────────────────
    // Use cudaMallocManaged for queue so host can inject tasks without memcpy
    CUDA_CHECK(cudaMallocManaged(&ctx->d_global_queue, sizeof_TaskQueue()));
    CUDA_CHECK(cudaMallocManaged(&ctx->d_inline_queue, sizeof_InlineTaskQueue()));

    CUDA_CHECK(cudaMalloc(&ctx->d_metrics,
        ctx->num_sms * METRIC_SLOT_SIZE));
    CUDA_CHECK(cudaMalloc(&ctx->d_sched_state,
        sizeof_SchedulerState()));
    CUDA_CHECK(cudaMemset(ctx->d_metrics, 0,
        ctx->num_sms * METRIC_SLOT_SIZE));
    CUDA_CHECK(cudaMemset(ctx->d_sched_state, 0,
        sizeof_SchedulerState()));

    // ── Pinned shutdown flag (host writes, GPU reads via UVA) ─────────────────
    CUDA_CHECK(cudaHostAlloc(&ctx->h_shutdown_flag, sizeof(uint32_t),
        cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(
        (void**)&ctx->d_shutdown_flag, ctx->h_shutdown_flag, 0));
    *ctx->h_shutdown_flag = 0;

    // ── Create dedicated stream for persistent kernel ──────────────────────────
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->kernel_stream,
        cudaStreamNonBlocking));

    printf("[Host] Microkernel context initialized.\n");
    printf("  Global queue: %p (%zu KB)\n",
        ctx->d_global_queue, sizeof_TaskQueue() / 1024);
    printf("  Payload pool: %zu slots\n", PAYLOAD_POOL_SLOTS);
    printf("\n");

    return ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL LAUNCH
// ─────────────────────────────────────────────────────────────────────────────

// Forward-declare the CUDA kernel (defined in microkernel.cu)
extern "C" void launch_gpu_microkernel(MicrokernelContext* ctx);

void microkernel_launch(MicrokernelContext* ctx) {
    printf("[Host] Launching persistent GPU microkernel...\n");
    printf("  Grid:  %d blocks (one per SM)\n", ctx->num_sms);
    printf("  Block: %d threads\n", ctx->block_threads);

    launch_gpu_microkernel(ctx);

    // Check launch didn't fail synchronously
    CUDA_CHECK(cudaGetLastError());
    printf("[Host] Microkernel launched. GPU is now autonomous.\n\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK INJECTION (Host → GPU Queue)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Inject a batch of tasks from the host into the managed queue.
 * Because d_global_queue is cudaMallocManaged, this write is visible to the
 * GPU without an explicit memcpy — the CUDA UVM system migrates pages.
 *
 * For latency-critical paths, use pinned memory + explicit async copies instead.
 */
void inject_tasks(MicrokernelContext* ctx,
                  const Task* tasks, int count,
                  TaskPayload* payloads) {
    // Copy payloads into the device-side pool
    CUDA_CHECK(cudaMemcpyAsync(
        // offset into sched_state->payload_pool
        (uint8_t*)ctx->d_sched_state + offsetof_payload_pool(),
        payloads,
        count * PAYLOAD_SIZE,
        cudaMemcpyHostToDevice,
        ctx->kernel_stream));

    // Enqueue task descriptors (managed memory, no explicit copy needed)
    // In production: use a lock-free ring from host side with UVM prefetching
    for (int i = 0; i < count; ++i) {
        uint32_t seg = i % 16;  // spread across segments
        bool ok = host_queue_enqueue(ctx->d_global_queue, seg, &tasks[i]);
        if (!ok) {
            fprintf(stderr, "[Host] WARNING: Queue full at task %d\n", i);
        }
    }

    // Prefetch queue pages to GPU to avoid demand-paging latency
    CUDA_CHECK(cudaMemAdvise(ctx->d_global_queue, sizeof_TaskQueue(),
        cudaMemAdviseSetPreferredLocation, 0 /* GPU device 0 */));
    CUDA_CHECK(cudaMemPrefetchAsync(ctx->d_global_queue, sizeof_TaskQueue(),
        0, ctx->kernel_stream));

    printf("[Host] Injected %d tasks.\n", count);
}

// ─────────────────────────────────────────────────────────────────────────────
// METRICS COLLECTION
// ─────────────────────────────────────────────────────────────────────────────

struct HostMetrics {
    unsigned long long total_completed;
    unsigned int       total_errors;
    unsigned int       total_watchdog_resets;
    int                active_sms;
};

HostMetrics collect_metrics(MicrokernelContext* ctx) {
    // Async copy metrics from device
    static uint8_t h_metrics_buf[108 * 64];
    CUDA_CHECK(cudaMemcpyAsync(h_metrics_buf,
        ctx->d_metrics,
        ctx->num_sms * METRIC_SLOT_SIZE,
        cudaMemcpyDeviceToHost,
        ctx->kernel_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->kernel_stream));

    HostMetrics result = {0, 0, 0, 0};
    for (int i = 0; i < ctx->num_sms; ++i) {
        // Parse metric slot at known offsets
        uint8_t* slot = h_metrics_buf + i * METRIC_SLOT_SIZE;
        unsigned long long completed;
        unsigned int errors, watchdog;
        memcpy(&completed, slot + 0,  8);
        memcpy(&errors,    slot + 8,  4);
        memcpy(&watchdog,  slot + 12, 4);

        result.total_completed     += completed;
        result.total_errors        += errors;
        result.total_watchdog_resets += watchdog;
        if (completed > 0) result.active_sms++;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// SHUTDOWN
// ─────────────────────────────────────────────────────────────────────────────

void microkernel_shutdown(MicrokernelContext* ctx) {
    printf("[Host] Signaling GPU microkernel shutdown...\n");

    // Atomic write to pinned memory — visible to GPU via UVA
    __atomic_store_n(ctx->h_shutdown_flag, 1, __ATOMIC_RELEASE);

    // Wait for kernel to finish (it exits when shutdown_flag is set)
    CUDA_CHECK(cudaStreamSynchronize(ctx->kernel_stream));

    HostMetrics m = collect_metrics(ctx);
    printf("[Host] Microkernel shutdown complete.\n");
    printf("  Total tasks completed:    %llu\n", m.total_completed);
    printf("  Total errors:             %u\n",   m.total_errors);
    printf("  Total watchdog resets:    %u\n",   m.total_watchdog_resets);
    printf("  Active SMs:               %d / %d\n", m.active_sms, ctx->num_sms);

    CUDA_CHECK(cudaFree(ctx->d_global_queue));
    CUDA_CHECK(cudaFree(ctx->d_inline_queue));
    CUDA_CHECK(cudaFree(ctx->d_metrics));
    CUDA_CHECK(cudaFree(ctx->d_sched_state));
    CUDA_CHECK(cudaFreeHost(ctx->h_shutdown_flag));
    CUDA_CHECK(cudaStreamDestroy(ctx->kernel_stream));
    delete ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN — Demo workflow
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("=== GPU Microkernel for Autonomous Scheduling ===\n\n");

    // 1. Initialize
    MicrokernelContext* ctx = microkernel_init();

    // 2. Launch persistent kernel (GPU is now autonomous)
    microkernel_launch(ctx);

    // 3. Inject initial workload
    // In a real system, this could be driven by network I/O, sensor data, etc.
    const int N_TASKS = 1000;
    static Task tasks[N_TASKS];
    static TaskPayload payloads[N_TASKS];

    for (int i = 0; i < N_TASKS; ++i) {
        tasks[i].type        = (i % 3 == 0) ? 0x02 : 0x01;  // REDUCE or COMPUTE
        tasks[i].priority    = i % 4;
        tasks[i].payload_idx = i;
        tasks[i].deadline    = 0;

        payloads[i].element_count = 4096;
        payloads[i].iterations    = 16;
        payloads[i].input_ptr     = 0;  // Would be real device pointer in production
        payloads[i].output_ptr    = 0;
        payloads[i].flags         = 0;
    }

    inject_tasks(ctx, tasks, N_TASKS, payloads);

    // 4. Host does other work while GPU runs autonomously
    printf("[Host] GPU running autonomously. Host free to do other work...\n");
    for (int t = 0; t < 5; ++t) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        HostMetrics m = collect_metrics(ctx);
        printf("[Host][t=%ds] completed=%llu, errors=%u, active_SMs=%d\n",
               t + 1, m.total_completed, m.total_errors, m.active_sms);
    }

    // 5. Shutdown
    microkernel_shutdown(ctx);
    return 0;
}
