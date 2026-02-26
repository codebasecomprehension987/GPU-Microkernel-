//! src/host.rs — Host-Side Control Plane (Rust)
//!
//! Uses the `cust` crate (idiomatic Rust CUDA bindings) to:
//!   1. Initialize the CUDA device
//!   2. Load the PTX module (compiled from lib.rs via rust-cuda)
//!   3. Manage device memory with RAII wrappers (auto-frees on drop)
//!   4. Launch kernels and collect results
//!
//! The C++ persistent microkernel is linked separately. This host binary
//! manages the control plane: injecting tasks and reading metrics.

use cust::prelude::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// DEVICE MEMORY WRAPPERS
// ─────────────────────────────────────────────────────────────────────────────

/// Typed device buffer. Freed automatically when dropped (RAII).
pub struct DeviceBuffer<T> {
    inner: cust::memory::DeviceBuffer<T>,
}

impl<T: cust::memory::DeviceCopy> DeviceBuffer<T> {
    pub fn zeroed(count: usize) -> cust::error::CudaResult<Self> {
        Ok(Self {
            inner: cust::memory::DeviceBuffer::zeroed(count)?,
        })
    }

    pub fn as_device_ptr(&self) -> cust::memory::DevicePointer<T> {
        self.inner.as_device_ptr()
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> cust::error::CudaResult<()> {
        self.inner.copy_from(src)
    }

    pub fn copy_to_host(&self, dst: &mut Vec<T>) -> cust::error::CudaResult<()>
    where T: Default + Clone
    {
        dst.resize(self.inner.len(), T::default());
        self.inner.copy_to(dst)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MICROKERNEL HOST INTERFACE
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, cust_core::DeviceCopy)]
pub struct MetricSlot {
    pub tasks_completed:  u64,
    pub errors:           u32,
    pub watchdog_resets:  u32,
    pub final_timestamp:  u64,
    pub _pad:             [u32; 4],
}

pub struct MicrokernelHost {
    _context:  cust::context::Context,
    stream:    cust::stream::Stream,
    num_sms:   u32,

    // Device allocations (freed on drop via RAII)
    d_metrics: DeviceBuffer<MetricSlot>,
    d_shutdown: cust::memory::UnifiedBuffer<u32>,

    // Shutdown coordination
    shutdown: Arc<AtomicBool>,
}

impl MicrokernelHost {
    pub fn new() -> cust::error::CudaResult<Self> {
        // Initialize CUDA (creates primary context)
        cust::init(cust::CudaFlags::empty())?;
        let device  = cust::device::Device::get_device(0)?;
        let context = cust::context::Context::new(device)?;

        println!("GPU: {}", device.name()?);
        println!("  Compute Capability: {}.{}",
            device.get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMajor)?,
            device.get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMinor)?);

        let num_sms = device.get_attribute(
            cust::device::DeviceAttribute::MultiprocessorCount)? as u32;
        println!("  SMs: {}", num_sms);

        let stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None)?;

        // Allocate metrics buffer
        let d_metrics = DeviceBuffer::zeroed(num_sms as usize)?;

        // Unified memory shutdown flag (visible to both host and device)
        let mut d_shutdown = cust::memory::UnifiedBuffer::new(&0u32, 1)?;

        Ok(Self {
            _context: context,
            stream,
            num_sms,
            d_metrics,
            d_shutdown,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Collect metrics from all SMs.
    pub fn collect_metrics(&mut self) -> cust::error::CudaResult<Vec<MetricSlot>> {
        let mut result = Vec::new();
        self.d_metrics.copy_to_host(&mut result)?;
        Ok(result)
    }

    /// Signal shutdown and wait for kernel to exit.
    pub fn shutdown(mut self) -> cust::error::CudaResult<()> {
        println!("[Host] Signaling GPU shutdown...");

        // Write to unified memory — visible to GPU without explicit sync
        unsafe {
            *self.d_shutdown.as_unified_ptr_mut() = 1u32;
        }

        // Sync the stream (waits for kernel to finish)
        self.stream.synchronize()?;

        // Print final metrics
        let metrics = self.collect_metrics()?;
        let total_completed: u64 = metrics.iter().map(|m| m.tasks_completed).sum();
        let total_errors:    u32 = metrics.iter().map(|m| m.errors).sum();
        let total_watchdog:  u32 = metrics.iter().map(|m| m.watchdog_resets).sum();
        let active_sms = metrics.iter().filter(|m| m.tasks_completed > 0).count();

        println!("[Host] Shutdown complete.");
        println!("  Tasks completed:    {}", total_completed);
        println!("  Errors:             {}", total_errors);
        println!("  Watchdog resets:    {}", total_watchdog);
        println!("  Active SMs:         {} / {}", active_sms, self.num_sms);

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK INJECTION — HOST SIDE
// ─────────────────────────────────────────────────────────────────────────────

/// Workload generator: produces a stream of tasks for the microkernel.
/// In production this would be fed by network I/O, sensor data, etc.
pub struct WorkloadGenerator {
    task_count: u64,
    rng_state:  u64,
}

impl WorkloadGenerator {
    pub fn new() -> Self {
        Self { task_count: 0, rng_state: 0xdeadbeef12345678 }
    }

    /// Simple xorshift64 RNG (no std::rand in embedded contexts)
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate a batch of tasks. Returns (task_type, element_count, priority).
    pub fn generate_batch(&mut self, count: usize) -> Vec<(u32, u32, u32)> {
        (0..count).map(|_| {
            let r = self.next_random();
            let task_type  = ((r & 0x3) + 1) as u32;  // 1, 2, or 3
            let elem_count = ((r >> 16) & 0x3FFF + 64) as u32;  // 64 - 16384
            let priority   = ((r >> 32) & 0x3) as u32;  // 0-3
            self.task_count += 1;
            (task_type, elem_count, priority)
        }).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MONITORING THREAD
// ─────────────────────────────────────────────────────────────────────────────

pub struct Monitor {
    interval: Duration,
    last_completed: u64,
    last_tick: Instant,
}

impl Monitor {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval:       Duration::from_secs(interval_secs),
            last_completed: 0,
            last_tick:      Instant::now(),
        }
    }

    pub fn tick(&mut self, metrics: &[MetricSlot]) {
        if self.last_tick.elapsed() < self.interval { return; }

        let total: u64 = metrics.iter().map(|m| m.tasks_completed).sum();
        let delta = total - self.last_completed;
        let elapsed = self.last_tick.elapsed().as_secs_f64();
        let throughput = delta as f64 / elapsed;

        println!("[Monitor] Throughput: {:.1} tasks/s | Total: {} | Errors: {}",
            throughput,
            total,
            metrics.iter().map(|m| m.errors as u64).sum::<u64>());

        self.last_completed = total;
        self.last_tick      = Instant::now();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> cust::error::CudaResult<()> {
    println!("=== GPU Microkernel — Rust Host Control Plane ===\n");

    let mut host     = MicrokernelHost::new()?;
    let mut gen      = WorkloadGenerator::new();
    let mut monitor  = Monitor::new(1);

    // The C++ microkernel is launched here (linked at build time).
    // In a real build this would call an FFI function.
    // launch_persistent_kernel(&host);  // FFI call to C++ launcher

    println!("[Host] Persistent kernel launched. Entering control loop.\n");

    // Control loop: inject tasks and monitor for 10 seconds
    let start = Instant::now();
    let run_duration = Duration::from_secs(10);

    while start.elapsed() < run_duration {
        // Generate and inject a batch of tasks
        let batch = gen.generate_batch(64);
        // In real implementation: serialize and write to managed queue
        // For demo, just log
        if start.elapsed().as_secs() % 2 == 0 {
            println!("[Host] Injected batch of {} tasks (total: {})",
                batch.len(), gen.task_count);
        }

        // Collect and display metrics
        let metrics = host.collect_metrics()?;
        monitor.tick(&metrics);

        std::thread::sleep(Duration::from_millis(100));
    }

    // Graceful shutdown
    host.shutdown()?;

    println!("\nDone.");
    Ok(())
}
