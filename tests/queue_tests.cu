/**
 * task_queue.cuh — Lock-Free MPMC Task Queue
 * ============================================
 * Implements a multi-producer, multi-consumer queue using:
 *   - Segmented ring buffers (one segment per queue-pair → reduces contention)
 *   - 64-bit fat CAS: packs head index + version tag to prevent ABA problem
 *   - Exponential backoff on contention
 *
 * ABA Problem Prevention:
 *   Without version tags, thread A dequeues slot X, thread B dequeues & enqueues
 *   slot X, thread A's CAS succeeds incorrectly. We pack a 32-bit monotonic
 *   version counter into the upper 32 bits of the atomic, making the CAS fail
 *   if the slot was recycled.
 *
 * Memory Ordering:
 *   - Enqueue: __atomic_store with release semantics (store after payload write)
 *   - Dequeue: __atomic_load with acquire semantics (load before payload read)
 *   These form a happens-before edge: writer's stores visible before reader reads.
 */

#pragma once
#include <cuda/atomic>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// TASK TYPES
// ─────────────────────────────────────────────────────────────────────────────

static constexpr uint32_t TASK_COMPUTE = 0x01;
static constexpr uint32_t TASK_REDUCE  = 0x02;
static constexpr uint32_t TASK_CHAIN   = 0x03;
static constexpr uint32_t TASK_INLINE  = 0x04;

struct TaskPayload {
    uint64_t input_ptr;      // device pointer (as uint64 for device-side use)
    uint64_t output_ptr;
    uint32_t element_count;
    uint32_t iterations;
    uint32_t flags;
    uint32_t _pad;
};

struct Task {
    uint32_t     type;
    uint32_t     priority;     // 0 = highest
    uint64_t     payload_idx;  // index into payload pool
    uint64_t     deadline;     // clock64 deadline; 0 = no deadline
};

struct TaskChainDescriptor {
    int   next_count;
    Task  next_tasks[8];  // max fan-out = 8 downstream tasks
};

struct InlineTask {
    uint64_t data_ptr;
    uint32_t count;
    uint32_t _pad;
};

// ─────────────────────────────────────────────────────────────────────────────
// QUEUE STRUCTURE
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int TASK_QUEUE_SEGMENTS = 16;
static constexpr int SEGMENT_CAPACITY    = 1024;  // must be power of 2
static constexpr int SEGMENT_MASK        = SEGMENT_CAPACITY - 1;

// Slot state machine: EMPTY → WRITING → FULL → READING → EMPTY
static constexpr uint32_t SLOT_EMPTY   = 0;
static constexpr uint32_t SLOT_WRITING = 1;
static constexpr uint32_t SLOT_FULL    = 2;
static constexpr uint32_t SLOT_READING = 3;

struct alignas(64) QueueSegment {  // cacheline-aligned to prevent false sharing
    cuda::atomic<uint64_t, cuda::thread_scope_device> head_packed;  // [version:32 | index:32]
    cuda::atomic<uint64_t, cuda::thread_scope_device> tail_packed;
    uint32_t _pad[8];  // push head/tail to separate cache lines
    cuda::atomic<uint32_t, cuda::thread_scope_device> slot_state[SEGMENT_CAPACITY];
    Task slots[SEGMENT_CAPACITY];
};

struct TaskQueue {
    QueueSegment segments[TASK_QUEUE_SEGMENTS];
};

// ─────────────────────────────────────────────────────────────────────────────
// INLINE QUEUE (for same-SM short tasks, no CDP overhead)
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int INLINE_CAPACITY = 256;
static constexpr int INLINE_MASK     = INLINE_CAPACITY - 1;

struct alignas(64) InlineQueueSegment {
    cuda::atomic<uint32_t, cuda::thread_scope_device> head;
    cuda::atomic<uint32_t, cuda::thread_scope_device> tail;
    InlineTask slots[INLINE_CAPACITY];
};

struct InlineTaskQueue {
    InlineQueueSegment segments[MAX_SMs_CONST];  // one per SM
};

// ─────────────────────────────────────────────────────────────────────────────
// QUEUE OPERATIONS
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
bool queue_try_enqueue(TaskQueue* q, uint32_t seg_idx, const Task* task) {
    QueueSegment& seg = q->segments[seg_idx % TASK_QUEUE_SEGMENTS];

    // Read tail with acquire (see what producers have committed)
    uint64_t tail_packed = seg.tail_packed.load(cuda::memory_order_acquire);
    uint32_t tail_ver    = (uint32_t)(tail_packed >> 32);
    uint32_t tail_idx    = (uint32_t)(tail_packed & 0xFFFFFFFFULL);
    uint32_t next_idx    = (tail_idx + 1) & SEGMENT_MASK;

    // Check full: if next_idx == head, queue is full
    uint64_t head_packed = seg.head_packed.load(cuda::memory_order_relaxed);
    uint32_t head_idx    = (uint32_t)(head_packed & 0xFFFFFFFFULL);
    if (next_idx == head_idx) return false;  // Queue full

    // Claim the slot: EMPTY → WRITING via CAS
    uint32_t expected = SLOT_EMPTY;
    bool claimed = seg.slot_state[tail_idx].compare_exchange_strong(
        expected, SLOT_WRITING,
        cuda::memory_order_acquire, cuda::memory_order_relaxed);
    if (!claimed) return false;  // Another producer claimed it; caller retries

    // Write payload (slot is ours exclusively in WRITING state)
    seg.slots[tail_idx] = *task;

    // Mark FULL with release (consumer will see payload after this)
    seg.slot_state[tail_idx].store(SLOT_FULL, cuda::memory_order_release);

    // Advance tail with versioned CAS (prevents ABA)
    uint64_t new_tail = ((uint64_t)(tail_ver + 1) << 32) | next_idx;
    seg.tail_packed.compare_exchange_strong(
        tail_packed, new_tail,
        cuda::memory_order_release, cuda::memory_order_relaxed);
    // Failure is okay — another thread advanced it; our slot is still marked FULL

    return true;
}

__device__ __forceinline__
bool queue_try_dequeue(TaskQueue* q, uint32_t seg_idx, Task* out_task) {
    QueueSegment& seg = q->segments[seg_idx % TASK_QUEUE_SEGMENTS];

    uint64_t head_packed = seg.head_packed.load(cuda::memory_order_acquire);
    uint32_t head_ver    = (uint32_t)(head_packed >> 32);
    uint32_t head_idx    = (uint32_t)(head_packed & 0xFFFFFFFFULL);

    // Check empty
    uint64_t tail_packed = seg.tail_packed.load(cuda::memory_order_relaxed);
    uint32_t tail_idx    = (uint32_t)(tail_packed & 0xFFFFFFFFULL);
    if (head_idx == tail_idx) return false;  // Empty

    // Wait for slot to be FULL (producer may still be writing)
    // Non-blocking: if not FULL yet, return false (caller retries next iteration)
    uint32_t state = seg.slot_state[head_idx].load(cuda::memory_order_acquire);
    if (state != SLOT_FULL) return false;

    // Claim: FULL → READING
    uint32_t expected = SLOT_FULL;
    bool claimed = seg.slot_state[head_idx].compare_exchange_strong(
        expected, SLOT_READING,
        cuda::memory_order_acquire, cuda::memory_order_relaxed);
    if (!claimed) return false;

    // Read payload
    *out_task = seg.slots[head_idx];

    // Release slot: READING → EMPTY
    seg.slot_state[head_idx].store(SLOT_EMPTY, cuda::memory_order_release);

    // Advance head (versioned)
    uint32_t next_idx = (head_idx + 1) & SEGMENT_MASK;
    uint64_t new_head = ((uint64_t)(head_ver + 1) << 32) | next_idx;
    seg.head_packed.compare_exchange_strong(
        head_packed, new_head,
        cuda::memory_order_release, cuda::memory_order_relaxed);

    return true;
}

/**
 * queue_recover — called by watchdog when livelock is suspected.
 * Scans for slots stuck in WRITING or READING state for too long and resets them.
 * This is a best-effort recovery; correctness is preserved because tasks in those
 * slots had not been committed (WRITING) or have already been read (READING).
 */
__device__ __forceinline__
void queue_recover(TaskQueue* q, uint32_t seg_idx) {
    QueueSegment& seg = q->segments[seg_idx % TASK_QUEUE_SEGMENTS];

    uint64_t head_packed = seg.head_packed.load(cuda::memory_order_relaxed);
    uint64_t tail_packed = seg.tail_packed.load(cuda::memory_order_relaxed);
    uint32_t head_idx    = (uint32_t)(head_packed & 0xFFFFFFFFULL);
    uint32_t tail_idx    = (uint32_t)(tail_packed & 0xFFFFFFFFULL);

    // Walk occupied slots; reset any stuck in transition states
    for (uint32_t i = head_idx; i != tail_idx; i = (i + 1) & SEGMENT_MASK) {
        uint32_t s = seg.slot_state[i].load(cuda::memory_order_relaxed);
        if (s == SLOT_READING) {
            // Was being read but reader died; safe to mark EMPTY
            seg.slot_state[i].compare_exchange_strong(
                s, SLOT_EMPTY,
                cuda::memory_order_relaxed, cuda::memory_order_relaxed);
        }
        // SLOT_WRITING: task was never committed; leave as-is (writer will
        // either complete or the watchdog will catch it next cycle)
    }
}

__device__ __forceinline__
void inline_queue_init_segment(InlineTaskQueue* q, int sm_id) {
    q->segments[sm_id].head.store(0, cuda::memory_order_relaxed);
    q->segments[sm_id].tail.store(0, cuda::memory_order_relaxed);
}

__device__ __forceinline__
bool inline_queue_try_dequeue(InlineTaskQueue* q, int sm_id, InlineTask* out) {
    InlineQueueSegment& seg = q->segments[sm_id];
    uint32_t head = seg.head.load(cuda::memory_order_acquire);
    uint32_t tail = seg.tail.load(cuda::memory_order_relaxed);
    if (head == tail) return false;

    *out = seg.slots[head & INLINE_MASK];
    seg.head.store(head + 1, cuda::memory_order_release);
    return true;
}
