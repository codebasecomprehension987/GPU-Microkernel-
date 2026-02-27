# Profiling the GPU Microkernel with Nsight Compute

## Quick Start

```bash
# Full profile (all metrics — slow but comprehensive)
ncu --set full \
    --target-processes all \
    --import-source yes \
    --export profile_report \
    ./microkernel

# Open in Nsight Compute GUI
ncu-ui profile_report.ncu-rep
```

## Key Metrics to Monitor

### Scheduler Health

| Metric | ncu Name | Healthy Range | Problem if... |
|--------|----------|---------------|---------------|
| Active warps/SM | `sm__warps_active.avg` | 28-32 | <16: register spill or launch bound |
| Warp stall (no instruction) | `sm__warps_eligible.avg` | >24 | <16: memory latency hiding poor |
| CDP launch count | `device__cuda_dynamic_parallelism_launch_count` | >0 | 0: no tasks dispatched |
| L2 atomic hit rate | `l2_global_atomic_store` | — | High: queue contention |

### Queue Contention

```bash
# Measure atomic pressure on queue head/tail
ncu --metrics \
    l2_global_atomic_store,\
    l2_global_atomic_load,\
    sm__sass_inst_executed_op_atom_dot_alu.sum \
    --kernel-name gpu_microkernel \
    ./microkernel
```

High `l2_global_atomic_store` relative to task count means queue contention.
Fix: increase `TASK_QUEUE_SEGMENTS` or add backoff between CAS retries.

### Memory Bandwidth

```bash
ncu --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l2__read_hit_rate.pct,\
    l2__write_hit_rate.pct \
    --kernel-name gpu_microkernel \
    ./microkernel
```

The task queue should stay in L2 (hit rate >80%). If it spills to DRAM:
- Reduce `TASK_QUEUE_SEGMENTS` (smaller working set)
- Add `__prefetch()` hints for next-to-dequeue slots

### CDP Child Kernel Profiling

CDP child kernels appear as separate kernels in ncu. Profile them with:

```bash
ncu --kernel-name child_compute_kernel ./microkernel
ncu --kernel-name child_reduce_kernel ./microkernel
```

## Common Pathologies and Fixes

### Low Occupancy (<16 active warps/SM)

Cause: Register spill (compiler using local memory because registers exhausted)

```bash
# Check register usage
ncu --metrics sm__warps_active.avg,l1tex__lsq_requests_op_ld_dot_addrgen_pipe_lsul.sum \
    --kernel-name gpu_microkernel ./microkernel

# Inspect PTX register count
cuobjdump --dump-ptx microkernel | grep "\.reg"
```

Fix: Lower `-maxrregcount` (try 64 → 48 → 32, check for spill in ptxas output)

### Watchdog Firing Repeatedly

Cause: Queue segment imbalance (one segment overloaded, others empty)

```bash
# Check per-SM metrics
ncu --metrics sm__warps_active.avg --target-processes all ./microkernel
# Look for variance across SMs
```

Fix: Change segment assignment to use `(blockIdx.x + warp_id) % SEGMENTS`
instead of `blockIdx.x % SEGMENTS` to improve distribution.

### CDP Launch Stalls

Cause: Too many concurrent CDP launches saturating the device command processor

```bash
ncu --metrics \
    device__cuda_dynamic_parallelism_launch_count,\
    sm__warps_eligible.avg.per_cycle_elapsed \
    --kernel-name gpu_microkernel ./microkernel
```

Fix: Batch small tasks into single larger child kernels.
Instead of launching N kernels of 32 threads, launch 1 kernel of N×32 threads.

### High UVM Demand-Paging Latency

If `inject_tasks()` is slow and the GPU stalls waiting for pages:

```bash
# Check UVM migration events
nvprof --print-gpu-trace --unified-memory-profiling per-process-device ./microkernel
```

Fix: Add `cudaMemPrefetchAsync` immediately after host writes to managed memory.

## Full Profiling Commands Reference

```bash
# Memory checker (catches OOB, use-after-free in device code)
compute-sanitizer --tool memcheck --check-device-heap yes ./microkernel

# Race condition detector (slower; catches queue race conditions)
compute-sanitizer --tool racecheck ./microkernel

# Deadlock detector
compute-sanitizer --tool synccheck ./microkernel

# Complete Nsight Systems timeline (host + device + UVM events)
nsys profile \
    --trace=cuda,cudnn,nvtx,osrt \
    --cuda-um-gpu-page-faults=true \
    --cuda-um-cpu-page-faults=true \
    --output=timeline \
    ./microkernel
nsys-ui timeline.qdrep
```

## Interpreting Warp Specialization in the Profiler

ncu will show `gpu_microkernel` with mixed utilization profiles because
scheduler warps and worker warps have completely different instruction mixes:

- Scheduler warps: high atomic instruction ratio, low FP throughput
- Worker warps: high FP throughput, low atomic ratio

The aggregate numbers will look "wrong" — e.g., FP utilization appears low
even though worker warps are compute-bound. This is expected and correct.

Use `ncu --section WarpStateStats` to see per-warp-state breakdown and
confirm both warp types are making forward progress.
