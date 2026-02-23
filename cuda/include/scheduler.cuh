/**
 * scheduler.cuh — Scheduler State Structures
 * ============================================
 */

#pragma once
#include <cuda/atomic>
#include "task_queue.cuh"

static constexpr int MAX_SMs_CONST   = 108;
static constexpr int PAYLOAD_POOL_SZ = 65536;

struct SchedulerState {
    cuda::atomic<uint64_t, cuda::thread_scope_device> tasks_dispatched;
    cuda::atomic<uint64_t, cuda::thread_scope_device> tasks_enqueued;
    uint32_t                                           flags;
    uint32_t                                           _pad[3];
    TaskPayload                                        payload_pool[PAYLOAD_POOL_SZ];
};
