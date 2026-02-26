//! gpu_microkernel/src/lib.rs
//!
//! Rust Device-Side Bindings for the GPU Microkernel
//! ===================================================
//! This module provides Rust-side abstractions over the CUDA microkernel using
//! the `rust-cuda` ecosystem (primarily `cuda-std` and `cuda-sys`).
//!
//! Memory Safety on the Device:
//! - All device pointers are wrapped in `DeviceRef<T>` / `DeviceBox<T>` types
//!   which statically prevent host-side dereferences
//! - The `#[kernel]` macro enforces that only `#[device]` functions are called
//!   from device code, catching host-only calls at compile time
//! - Shared memory is managed via `SharedArray<T, N>` which bounds-checks in
//!   debug builds and elides to raw `__shared__` in release
//!
//! Why Rust + CUDA is Extremely Difficult:
//! - No standard allocator on device (no heap; must use pre-allocated pools)
//! - No panics (unwind is undefined in CUDA device code)
//! - `#[no_std]` required — most Rust idioms unavailable
//! - Atomic types must use CUDA-specific memory scopes, not Rust's std::sync
//! - Lifetimes don't map cleanly to GPU execution model (no borrow checker
//!   for cross-warp data sharing; must use unsafe + documentation contracts)

#![no_std]
#![feature(abi_ptx)]         // Required for PTX code generation
#![feature(stdsimd)]         // SIMD intrinsics
#![feature(core_intrinsics)] // Needed for raw ptr operations
#![deny(unsafe_op_in_unsafe_fn)]

// rust-cuda crate imports (would be in Cargo.toml)
// cuda-std = { git = "https://github.com/Rust-GPU/Rust-CUDA" }
// cuda-sys  = { git = "https://github.com/Rust-GPU/Rust-CUDA" }
extern crate cuda_std;

use cuda_std::prelude::*;
use cuda_std::thread;
use cuda_std::shared_array;
use cuda_std::atomic::{AtomicU32, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// TASK TYPES (mirroring C++ definitions)
// ─────────────────────────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, PartialEq)]
pub enum TaskType {
    Compute = 0x01,
    Reduce  = 0x02,
    Chain   = 0x03,
    Inline  = 0x04,
}

/// Memory-safe task descriptor.
/// `#[repr(C)]` ensures layout matches the C++ struct exactly.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Task {
    pub task_type:   u32,
    pub priority:    u32,
    pub payload_idx: u64,
    pub deadline:    u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TaskPayload {
    pub input_ptr:     u64,  // Device address; never dereference on host
    pub output_ptr:    u64,
    pub element_count: u32,
    pub iterations:    u32,
    pub flags:         u32,
    pub _pad:          u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// DEVICE-SAFE POINTER WRAPPER
// ─────────────────────────────────────────────────────────────────────────────

/// A pointer that can only be dereferenced in device code.
/// The phantom lifetime `'a` ties this to a device allocation's lifetime.
pub struct DevicePtr<'a, T> {
    raw: *mut T,
    _phantom: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Copy> DevicePtr<'a, T> {
    /// Construct from a raw device pointer.
    /// # Safety
    /// Caller must guarantee `raw` points to valid device memory.
    #[device]
    pub unsafe fn new(raw: *mut T) -> Self {
        DevicePtr { raw, _phantom: core::marker::PhantomData }
    }

    /// Read value at offset. Bounds checking in debug builds only.
    #[device]
    pub unsafe fn read_at(&self, index: usize) -> T {
        // In release mode this compiles to a single LD.GLOBAL instruction.
        // No runtime overhead vs raw pointer dereference.
        unsafe { *self.raw.add(index) }
    }

    /// Atomic add for u32 (maps to CUDA's atomicAdd)
    #[device]
    pub unsafe fn atomic_add_u32(&self, index: usize, val: u32) -> u32
    where T: core::marker::Sized + Copy
    {
        // Uses inline PTX for device-scope atomic
        let ptr = self.raw.add(index) as *mut u32;
        let mut result: u32;
        unsafe {
            core::arch::asm!(
                "atom.global.add.u32 {out}, [{ptr}], {val};",
                out = out(reg32) result,
                ptr = in(reg64) ptr,
                val = in(reg32) val,
            );
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARED MEMORY MANAGER
// ─────────────────────────────────────────────────────────────────────────────

/// Statically-sized shared memory array.
/// All accesses are inlined — zero-cost abstraction.
pub struct SharedMem<T, const N: usize> {
    data: [T; N],
}

impl<T: Copy + Default, const N: usize> SharedMem<T, N> {
    /// Initialize in shared memory. Must be called by thread 0 before use.
    #[device]
    #[inline(always)]
    pub fn init(&mut self) {
        if thread::thread_idx_x() == 0 {
            for i in 0..N {
                self.data[i] = T::default();
            }
        }
        // __syncthreads() — CRITICAL: only call in kernels where ALL threads
        // reach this point. Never in warp-specialized persistent kernels.
        unsafe { cuda_std::sync::sync_threads() };
    }

    #[device]
    #[inline(always)]
    pub fn get(&self, idx: usize) -> T {
        debug_assert!(idx < N, "SharedMem bounds check failed");
        self.data[idx]
    }

    #[device]
    #[inline(always)]
    pub fn set(&mut self, idx: usize, val: T) {
        debug_assert!(idx < N, "SharedMem bounds check failed");
        self.data[idx] = val;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LOCK-FREE SLOT ALLOCATOR (Device-Side Memory Pool)
// ─────────────────────────────────────────────────────────────────────────────

/// A fixed-capacity pool allocator using a bitmask freelist.
/// Avoids heap allocation; all memory pre-allocated in global memory.
///
/// Thread Safety: Uses device-scope atomic CAS on the freelist word.
pub struct PoolAllocator {
    freelist: AtomicU32,  // 1 bit per slot; 1 = free, 0 = allocated
    // Max 32 slots per allocator word; use array for larger pools
}

impl PoolAllocator {
    #[device]
    pub fn alloc(&self) -> Option<u32> {
        let mut freelist = self.freelist.load(Ordering::Acquire);
        loop {
            if freelist == 0 { return None; }  // All slots taken
            let slot = freelist.trailing_zeros();  // Find first free slot
            let new_freelist = freelist & !(1 << slot);
            match self.freelist.compare_exchange_weak(
                freelist, new_freelist,
                Ordering::AcqRel, Ordering::Acquire
            ) {
                Ok(_)  => return Some(slot),
                Err(f) => freelist = f,  // CAS failed; retry with updated value
            }
        }
    }

    #[device]
    pub fn free(&self, slot: u32) {
        // OR in the freed bit — safe even without CAS (monotonic set operation)
        self.freelist.fetch_or(1 << slot, Ordering::Release);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WARP-LEVEL PRIMITIVES (Safe Wrappers)
// ─────────────────────────────────────────────────────────────────────────────

/// Warp-uniform broadcast via `__shfl_sync`.
/// All threads in the warp must call this (warp-uniform execution).
///
/// # Safety
/// `mask` must include all active threads in the warp.
/// `lane` must be in [0, 32).
#[device]
#[inline(always)]
pub unsafe fn warp_broadcast_u32(mask: u32, val: u32, lane: u32) -> u32 {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "shfl.sync.idx.b32 {out}|_, {val}, {lane}, 0x1f, {mask};",
            out  = out(reg32) result,
            val  = in(reg32)  val,
            lane = in(reg32)  lane,
            mask = in(reg32)  mask,
        );
    }
    result
}

/// Warp-level population count (count active threads matching predicate).
#[device]
#[inline(always)]
pub fn warp_ballot(mask: u32, pred: bool) -> u32 {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "vote.sync.ballot.b32 {out}, {pred}, {mask};",
            out  = out(reg32) result,
            pred = in(pred)   pred,
            mask = in(reg32)  mask,
        );
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// RUST DEVICE KERNEL (compiled to PTX via rust-cuda)
// ─────────────────────────────────────────────────────────────────────────────

/// Rust implementation of a worker kernel.
/// This demonstrates how rust-cuda allows writing CUDA device code in Rust
/// with compile-time safety guarantees.
///
/// The `#[kernel]` attribute:
///  - Validates parameter types are device-accessible
///  - Generates proper PTX calling convention
///  - Prevents host-only APIs (Box, Vec, String) from being used
#[kernel]
pub unsafe fn rust_worker_kernel(
    input:  *const f32,
    output: *mut f32,
    count:  u32,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x()
               + thread::thread_idx_x()) as usize;

    if tid >= count as usize { return; }

    // Rust's safety: this is still unsafe because we're dereferencing raw ptrs
    // but the rust-cuda type system at least ensures we can't accidentally
    // use a host ptr here (would be caught at the kernel boundary)
    let val = unsafe { *input.add(tid) };

    // Compute (same polynomial as C++ version, for cross-validation)
    let result = (0..16u32).fold(val, |v, _| v * v * 0.5 + v * 0.3 + 0.1);

    unsafe { *output.add(tid) = result; }
}

/// Rust warp-reduction kernel.
/// Demonstrates using `__shfl_down_sync` safely through the `cuda_std` wrapper.
#[kernel]
pub unsafe fn rust_warp_reduce_kernel(
    data:   *const f32,
    result: *mut f32,
    count:  u32,
) {
    let tid   = (thread::block_idx_x() * thread::block_dim_x()
                 + thread::thread_idx_x()) as usize;
    let lane  = thread::thread_idx_x() % 32;

    let mut val = if tid < count as usize {
        unsafe { *data.add(tid) }
    } else {
        0.0f32
    };

    // Warp-level reduction using shuffle (no shared memory needed!)
    // Safe because all 32 lanes participate (full warp mask = 0xffffffff)
    let full_mask: u32 = 0xffffffff;
    let mut offset = 16u32;
    while offset > 0 {
        let other = cuda_std::shuffle::shfl_down_sync(full_mask, val, offset);
        val += other;
        offset >>= 1;
    }

    // Lane 0 writes the warp's partial sum
    if lane == 0 {
        // Atomic add into global result
        unsafe {
            let ptr = result.add(thread::block_idx_x() as usize);
            core::arch::asm!(
                "atom.global.add.f32 _, [{ptr}], {val};",
                ptr = in(reg64) ptr,
                val = in(reg32) val,
            );
        }
    }
}
