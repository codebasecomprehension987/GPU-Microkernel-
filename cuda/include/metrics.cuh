/**
 * metrics.cuh — Per-SM Performance Counters
 */

#pragma once
#include <stdint.h>

struct alignas(64) MetricSlot {
    unsigned long long tasks_completed;
    unsigned int       errors;
    unsigned int       watchdog_resets;
    unsigned long long final_timestamp;
    uint32_t           _pad[4];
};
