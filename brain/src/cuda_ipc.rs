//! CUDA IPC (Inter-Process Communication) for multi-process pipeline parallelism.
//!
//! Replaces NCCL for same-machine multi-GPU training. Each process owns one GPU.
//! Activation tensors are exchanged between adjacent pipeline stages via shared
//! GPU memory handles — zero-copy NVLink access, no NCCL dependency.
//!
//! Protocol:
//!   1. Each process allocates send/recv buffers on its GPU
//!   2. IPC memory handles are exchanged via files in a shared directory
//!   3. Forward: process i writes activations to its send buffer,
//!      process i+1 reads them via the mapped IPC pointer
//!   4. Synchronization via CUDA IPC events (also exchanged as handles)

use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

// ─── CUDA IPC types ──────────────────────────────────────────────────────────

/// cudaIpcMemHandle_t — 64-byte opaque struct
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CudaIpcMemHandle {
    pub reserved: [u8; 64],
}

/// cudaIpcEventHandle_t — 64-byte opaque struct
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CudaIpcEventHandle {
    pub reserved: [u8; 64],
}

pub type CudaStream = *mut c_void;
pub type CudaEvent = *mut c_void;

/// cudaIpcMemLazyEnablePeerAccess = 0x01
pub const CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 0x01;

const CUDA_SUCCESS: i32 = 0;

// ─── CUDA Runtime FFI ────────────────────────────────────────────────────────

unsafe extern "C" {
    // Memory allocation
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemsetAsync(devPtr: *mut c_void, value: i32, count: usize, stream: CudaStream) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void, src: *const c_void, count: usize,
        kind: i32, stream: CudaStream,
    ) -> i32;

    // IPC memory
    fn cudaIpcGetMemHandle(handle: *mut CudaIpcMemHandle, devPtr: *mut c_void) -> i32;
    fn cudaIpcOpenMemHandle(
        devPtr: *mut *mut c_void,
        handle: CudaIpcMemHandle,
        flags: u32,
    ) -> i32;
    fn cudaIpcCloseMemHandle(devPtr: *mut c_void) -> i32;

    // IPC events
    fn cudaIpcGetEventHandle(handle: *mut CudaIpcEventHandle, event: CudaEvent) -> i32;
    fn cudaIpcOpenEventHandle(event: *mut CudaEvent, handle: CudaIpcEventHandle) -> i32;

    // Events
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventCreateWithFlags(event: *mut CudaEvent, flags: u32) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> i32;
    fn cudaStreamWaitEvent(stream: CudaStream, event: CudaEvent, flags: u32) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;

    // Device management
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaStreamSynchronize(stream: CudaStream) -> i32;
    fn cudaStreamCreate(pStream: *mut CudaStream) -> i32;
    fn cudaStreamDestroy(stream: CudaStream) -> i32;

    fn cudaGetErrorString(error: i32) -> *const i8;
}

/// cudaEventInterprocess | cudaEventDisableTiming
const EVENT_IPC_FLAGS: u32 = 0x04 | 0x02;

/// cudaMemcpyDeviceToDevice
const MEMCPY_D2D: i32 = 3;

// ─── Error handling ──────────────────────────────────────────────────────────

fn check_cuda(result: i32, op: &str) {
    if result != CUDA_SUCCESS {
        let msg = unsafe {
            let ptr = cudaGetErrorString(result);
            if ptr.is_null() {
                format!("error code {result}")
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        panic!("CUDA {op} failed: {msg}");
    }
}

// ─── IPC shared buffer ──────────────────────────────────────────────────────

/// A GPU buffer that can be shared across processes via CUDA IPC.
///
/// The owning process allocates the buffer and exports the IPC handle.
/// Remote processes import the handle to get a device pointer into the
/// same physical memory (zero-copy via NVLink/PCIe BAR).
pub struct IpcBuffer {
    /// Device pointer on the owning process's GPU
    pub ptr: *mut c_void,
    /// Size in bytes
    pub nbytes: usize,
    /// IPC memory handle (exported by owner)
    pub handle: CudaIpcMemHandle,
    /// Whether this process owns the allocation (and should cudaFree it)
    owned: bool,
}

impl IpcBuffer {
    /// Allocate a new GPU buffer and export its IPC handle.
    pub fn alloc(nbytes: usize) -> Self {
        let mut ptr: *mut c_void = ptr::null_mut();
        check_cuda(unsafe { cudaMalloc(&mut ptr, nbytes) }, "Malloc(IpcBuffer)");

        let mut handle = CudaIpcMemHandle { reserved: [0u8; 64] };
        check_cuda(
            unsafe { cudaIpcGetMemHandle(&mut handle, ptr) },
            "IpcGetMemHandle",
        );

        Self { ptr, nbytes, handle, owned: true }
    }

    /// Import a buffer from a remote process's IPC handle.
    /// The returned pointer maps to the same physical GPU memory.
    pub fn from_handle(handle: CudaIpcMemHandle) -> Self {
        let mut ptr: *mut c_void = ptr::null_mut();
        check_cuda(
            unsafe {
                cudaIpcOpenMemHandle(&mut ptr, handle, CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
            },
            "IpcOpenMemHandle",
        );

        Self { ptr, nbytes: 0, handle, owned: false }
    }

    pub fn zero(&self, stream: CudaStream) {
        if self.nbytes > 0 {
            check_cuda(
                unsafe { cudaMemsetAsync(self.ptr, 0, self.nbytes, stream) },
                "MemsetAsync(IpcBuffer)",
            );
        }
    }
}

impl Drop for IpcBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() { return; }
        if self.owned {
            unsafe { cudaFree(self.ptr) };
        } else {
            unsafe { cudaIpcCloseMemHandle(self.ptr) };
        }
    }
}

unsafe impl Send for IpcBuffer {}

// ─── IPC event (cross-process synchronization) ──────────────────────────────

/// A CUDA event that can be shared across processes for synchronization.
/// The producer records the event after writing; the consumer waits on it
/// before reading.
pub struct IpcEvent {
    pub event: CudaEvent,
    pub handle: CudaIpcEventHandle,
    owned: bool,
}

impl IpcEvent {
    /// Create a new IPC-capable event and export its handle.
    pub fn new() -> Self {
        let mut event: CudaEvent = ptr::null_mut();
        check_cuda(
            unsafe { cudaEventCreateWithFlags(&mut event, EVENT_IPC_FLAGS) },
            "EventCreateWithFlags(IPC)",
        );

        let mut handle = CudaIpcEventHandle { reserved: [0u8; 64] };
        check_cuda(
            unsafe { cudaIpcGetEventHandle(&mut handle, event) },
            "IpcGetEventHandle",
        );

        Self { event, handle, owned: true }
    }

    /// Import an event from a remote process's IPC handle.
    pub fn from_handle(handle: CudaIpcEventHandle) -> Self {
        let mut event: CudaEvent = ptr::null_mut();
        check_cuda(
            unsafe { cudaIpcOpenEventHandle(&mut event, handle) },
            "IpcOpenEventHandle",
        );

        Self { event, handle, owned: false }
    }

    /// Record this event on the given stream (producer side).
    pub fn record(&self, stream: CudaStream) {
        check_cuda(
            unsafe { cudaEventRecord(self.event, stream) },
            "EventRecord(IPC)",
        );
    }

    /// Make `stream` wait until this event is recorded (consumer side).
    pub fn wait(&self, stream: CudaStream) {
        check_cuda(
            unsafe { cudaStreamWaitEvent(stream, self.event, 0) },
            "StreamWaitEvent(IPC)",
        );
    }

    /// Block the CPU until the event completes.
    pub fn synchronize(&self) {
        check_cuda(
            unsafe { cudaEventSynchronize(self.event) },
            "EventSynchronize(IPC)",
        );
    }
}

impl Drop for IpcEvent {
    fn drop(&mut self) {
        if !self.event.is_null() && self.owned {
            unsafe { cudaEventDestroy(self.event) };
        }
    }
}

unsafe impl Send for IpcEvent {}

// ─── Handle exchange via filesystem ─────────────────────────────────────────

/// Serialized IPC handles for one pipeline stage, written to a file so
/// adjacent processes can discover each other.
#[repr(C)]
pub struct StageHandles {
    pub rank: u32,
    pub world_size: u32,
    /// IPC handle for this stage's send buffer (activations flowing forward)
    pub send_mem: CudaIpcMemHandle,
    /// IPC handle for this stage's recv buffer (activations received from prev)
    pub recv_mem: CudaIpcMemHandle,
    /// IPC event: signaled after this stage writes to its send buffer
    pub send_ready: CudaIpcEventHandle,
    /// IPC event: signaled after this stage finishes reading from recv buffer
    pub recv_done: CudaIpcEventHandle,
    /// IPC handle for gradient send buffer (backward direction)
    pub grad_send_mem: CudaIpcMemHandle,
    /// IPC handle for gradient recv buffer
    pub grad_recv_mem: CudaIpcMemHandle,
    /// IPC event: signaled after gradient write
    pub grad_send_ready: CudaIpcEventHandle,
    /// IPC event: signaled after gradient read
    pub grad_recv_done: CudaIpcEventHandle,
}

fn handle_path(dir: &Path, rank: u32) -> PathBuf {
    dir.join(format!("ipc_stage_{rank}.bin"))
}

impl StageHandles {
    /// Write handles to a file for other processes to read.
    pub fn write_to_file(&self, dir: &Path) {
        fs::create_dir_all(dir).expect("failed to create IPC handle dir");
        let path = handle_path(dir, self.rank);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        };
        fs::write(&path, bytes).unwrap_or_else(|e| {
            panic!("failed to write IPC handles to {}: {e}", path.display())
        });
    }

    /// Read handles from a file written by another process.
    /// Spins until the file appears and has the correct size.
    pub fn read_from_file(dir: &Path, rank: u32) -> Self {
        let path = handle_path(dir, rank);
        let expected = std::mem::size_of::<Self>();
        loop {
            if let Ok(data) = fs::read(&path) {
                if data.len() == expected {
                    let mut out = std::mem::MaybeUninit::<Self>::uninit();
                    unsafe {
                        ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            out.as_mut_ptr() as *mut u8,
                            expected,
                        );
                        return out.assume_init();
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

// ─── IPC Pipeline (replaces NcclPipeline for multi-process) ─────────────────

/// Multi-process pipeline stage communicator using CUDA IPC.
///
/// Each process creates one `IpcPipeline` for its rank. Handles are
/// exchanged via files so adjacent stages can directly access each
/// other's GPU memory through NVLink — zero-copy, no NCCL.
pub struct IpcPipeline {
    pub rank: u32,
    pub world_size: u32,
    pub stream: CudaStream,

    // ── Forward direction ────────────────────────────────────────────────
    /// Buffer this stage writes activations into (owned, allocated on our GPU)
    pub fwd_send: IpcBuffer,
    /// Pointer into the previous stage's send buffer (IPC-mapped, read-only)
    pub fwd_recv_remote: Option<*mut c_void>,
    /// Local recv buffer (owned) — copy target from remote pointer
    pub fwd_recv: IpcBuffer,

    /// Event: we record after writing fwd_send
    pub fwd_send_event: IpcEvent,
    /// Event: previous stage's send_ready (we wait on this before reading)
    pub fwd_prev_send_event: Option<IpcEvent>,
    /// Event: we record after finishing our read from prev's send buffer
    pub fwd_recv_done_event: IpcEvent,

    // ── Backward direction ───────────────────────────────────────────────
    /// Buffer this stage writes gradients into (owned)
    pub bwd_send: IpcBuffer,
    /// Pointer into the next stage's grad send buffer (IPC-mapped)
    pub bwd_recv_remote: Option<*mut c_void>,
    /// Local recv buffer for gradients (owned)
    pub bwd_recv: IpcBuffer,

    pub bwd_send_event: IpcEvent,
    pub bwd_next_send_event: Option<IpcEvent>,
    pub bwd_recv_done_event: IpcEvent,
}

impl IpcPipeline {
    /// Create a new IPC pipeline stage.
    ///
    /// `activation_bytes`: size of the activation tensor (e.g. BT * D_MODEL * sizeof(bf16))
    /// `handle_dir`: shared filesystem directory for IPC handle exchange
    ///
    /// All processes must call this concurrently. The function blocks until
    /// adjacent stages' handle files are available.
    pub fn new(
        rank: u32,
        world_size: u32,
        gpu_id: i32,
        activation_bytes: usize,
        handle_dir: &Path,
    ) -> Self {
        check_cuda(unsafe { cudaSetDevice(gpu_id) }, "SetDevice");

        let mut stream: CudaStream = ptr::null_mut();
        check_cuda(unsafe { cudaStreamCreate(&mut stream) }, "StreamCreate");

        // Allocate forward buffers
        let fwd_send = IpcBuffer::alloc(activation_bytes);
        let fwd_recv = IpcBuffer::alloc(activation_bytes);
        let fwd_send_event = IpcEvent::new();
        let fwd_recv_done_event = IpcEvent::new();

        // Allocate backward buffers
        let bwd_send = IpcBuffer::alloc(activation_bytes);
        let bwd_recv = IpcBuffer::alloc(activation_bytes);
        let bwd_send_event = IpcEvent::new();
        let bwd_recv_done_event = IpcEvent::new();

        // Publish our handles
        let my_handles = StageHandles {
            rank,
            world_size,
            send_mem: fwd_send.handle,
            recv_mem: fwd_recv.handle,
            send_ready: fwd_send_event.handle,
            recv_done: fwd_recv_done_event.handle,
            grad_send_mem: bwd_send.handle,
            grad_recv_mem: bwd_recv.handle,
            grad_send_ready: bwd_send_event.handle,
            grad_recv_done: bwd_recv_done_event.handle,
        };
        my_handles.write_to_file(handle_dir);

        // Open adjacent stages' handles
        let mut fwd_recv_remote = None;
        let mut fwd_prev_send_event = None;
        let mut bwd_recv_remote = None;
        let mut bwd_next_send_event = None;

        // Previous stage (forward recv)
        if rank > 0 {
            let prev = StageHandles::read_from_file(handle_dir, rank - 1);
            // Map previous stage's send buffer into our address space
            let remote = IpcBuffer::from_handle(prev.send_mem);
            fwd_recv_remote = Some(remote.ptr);
            // Don't drop — we need the mapping alive. Leak intentionally;
            // cleaned up when this process exits.
            std::mem::forget(remote);
            fwd_prev_send_event = Some(IpcEvent::from_handle(prev.send_ready));
        }

        // Next stage (backward recv)
        if rank + 1 < world_size {
            let next = StageHandles::read_from_file(handle_dir, rank + 1);
            let remote = IpcBuffer::from_handle(next.grad_send_mem);
            bwd_recv_remote = Some(remote.ptr);
            std::mem::forget(remote);
            bwd_next_send_event = Some(IpcEvent::from_handle(next.grad_send_ready));
        }

        Self {
            rank,
            world_size,
            stream,
            fwd_send,
            fwd_recv_remote,
            fwd_recv,
            fwd_send_event,
            fwd_prev_send_event,
            fwd_recv_done_event,
            bwd_send,
            bwd_recv_remote,
            bwd_recv,
            bwd_send_event,
            bwd_next_send_event,
            bwd_recv_done_event,
        }
    }

    // ─── Forward pass communication ──────────────────────────────────────

    /// Stage i: copy activations from `src_ptr` (our GPU) into the send buffer,
    /// then signal that the data is ready.
    pub fn fwd_send(&self, src_ptr: *const c_void, nbytes: usize) {
        check_cuda(
            unsafe {
                cudaMemcpyAsync(self.fwd_send.ptr, src_ptr, nbytes, MEMCPY_D2D, self.stream)
            },
            "fwd_send memcpy",
        );
        self.fwd_send_event.record(self.stream);
    }

    /// Stage i+1: wait for previous stage's send to complete, then copy from
    /// the IPC-mapped remote pointer into our local recv buffer.
    ///
    /// Returns the local recv buffer pointer (ready to use after stream sync).
    pub fn fwd_recv(&self, nbytes: usize) -> *const c_void {
        let prev_event = self.fwd_prev_send_event.as_ref()
            .expect("fwd_recv called on rank 0 (no previous stage)");
        let remote_ptr = self.fwd_recv_remote
            .expect("fwd_recv: no remote pointer mapped");

        // Wait until previous stage has finished writing
        prev_event.wait(self.stream);

        // Copy from remote GPU memory into our local buffer
        // (goes over NVLink — zero-copy from the remote GPU's perspective,
        // but we copy into local VRAM so subsequent kernels have local bandwidth)
        check_cuda(
            unsafe {
                cudaMemcpyAsync(
                    self.fwd_recv.ptr,
                    remote_ptr as *const c_void,
                    nbytes,
                    MEMCPY_D2D,
                    self.stream,
                )
            },
            "fwd_recv memcpy",
        );

        // Signal that we've finished reading (prev stage can reuse its send buffer)
        self.fwd_recv_done_event.record(self.stream);

        self.fwd_recv.ptr as *const c_void
    }

    // ─── Backward pass communication ─────────────────────────────────────

    /// Stage i: copy gradients into the backward send buffer, signal ready.
    pub fn bwd_send(&self, src_ptr: *const c_void, nbytes: usize) {
        check_cuda(
            unsafe {
                cudaMemcpyAsync(self.bwd_send.ptr, src_ptr, nbytes, MEMCPY_D2D, self.stream)
            },
            "bwd_send memcpy",
        );
        self.bwd_send_event.record(self.stream);
    }

    /// Stage i-1: wait for next stage's gradient send, copy into local buffer.
    pub fn bwd_recv(&self, nbytes: usize) -> *const c_void {
        let next_event = self.bwd_next_send_event.as_ref()
            .expect("bwd_recv called on last rank (no next stage)");
        let remote_ptr = self.bwd_recv_remote
            .expect("bwd_recv: no remote pointer mapped");

        next_event.wait(self.stream);

        check_cuda(
            unsafe {
                cudaMemcpyAsync(
                    self.bwd_recv.ptr,
                    remote_ptr as *const c_void,
                    nbytes,
                    MEMCPY_D2D,
                    self.stream,
                )
            },
            "bwd_recv memcpy",
        );

        self.bwd_recv_done_event.record(self.stream);

        self.bwd_recv.ptr as *const c_void
    }

    // ─── Utility ─────────────────────────────────────────────────────────

    pub fn synchronize(&self) {
        check_cuda(
            unsafe { cudaStreamSynchronize(self.stream) },
            "StreamSynchronize(IpcPipeline)",
        );
    }

    /// Clean up IPC handle files. Call from rank 0 after all processes are done.
    pub fn cleanup_handles(handle_dir: &Path, world_size: u32) {
        for r in 0..world_size {
            let _ = fs::remove_file(handle_path(handle_dir, r));
        }
    }
}

impl Drop for IpcPipeline {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}

unsafe impl Send for IpcPipeline {}
