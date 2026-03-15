use std::ffi::c_void;
use std::fs;
use std::path::Path;

// NCCL types
pub type NcclComm = *mut c_void;
pub type NcclUniqueId = [u8; 128]; // NCCL_UNIQUE_ID_BYTES = 128
pub type CUstream = *mut c_void;

// NCCL data types
pub const NCCL_FLOAT16: i32 = 6;
pub const NCCL_BFLOAT16: i32 = 6;

// ncclResult_t success
const NCCL_SUCCESS: i32 = 0;

unsafe extern "C" {
    fn ncclGetUniqueId(uniqueId: *mut NcclUniqueId) -> i32;
    fn ncclCommInitRank(comm: *mut NcclComm, nranks: i32, commId: NcclUniqueId, rank: i32) -> i32;
    fn ncclCommDestroy(comm: NcclComm) -> i32;
    fn ncclSend(sendbuff: *const c_void, count: usize, datatype: i32, peer: i32, comm: NcclComm, stream: CUstream) -> i32;
    fn ncclRecv(recvbuff: *mut c_void, count: usize, datatype: i32, peer: i32, comm: NcclComm, stream: CUstream) -> i32;
    fn ncclGroupStart() -> i32;
    fn ncclGroupEnd() -> i32;
}

fn check_nccl(result: i32, op: &str) {
    if result != NCCL_SUCCESS {
        panic!("NCCL {op} failed with error code {result}");
    }
}

pub struct NcclPipeline {
    comm: NcclComm,
    pub rank: i32,
    pub world_size: i32,
}

impl NcclPipeline {
    /// Init NCCL communicator. Rank 0 generates unique ID and writes to file;
    /// other ranks read it.
    pub fn new(rank: i32, world_size: i32, unique_id_file: &Path) -> Self {
        let mut unique_id: NcclUniqueId = [0u8; 128];

        if rank == 0 {
            let ret = unsafe { ncclGetUniqueId(&mut unique_id) };
            check_nccl(ret, "GetUniqueId");
            fs::write(unique_id_file, &unique_id).expect("failed to write NCCL unique ID file");
        } else {
            // Spin until rank 0 writes the file
            loop {
                if let Ok(data) = fs::read(unique_id_file) {
                    if data.len() == 128 {
                        unique_id.copy_from_slice(&data);
                        break;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }

        let mut comm: NcclComm = std::ptr::null_mut();
        let ret = unsafe { ncclCommInitRank(&mut comm, world_size, unique_id, rank) };
        check_nccl(ret, "CommInitRank");

        Self { comm, rank, world_size }
    }

    pub fn send_bf16(&self, buf_ptr: *const c_void, count: usize, dst_rank: i32, stream: CUstream) {
        let ret = unsafe { ncclSend(buf_ptr, count, NCCL_BFLOAT16, dst_rank, self.comm, stream) };
        check_nccl(ret, "Send");
    }

    pub fn recv_bf16(&self, buf_ptr: *mut c_void, count: usize, src_rank: i32, stream: CUstream) {
        let ret = unsafe { ncclRecv(buf_ptr, count, NCCL_BFLOAT16, src_rank, self.comm, stream) };
        check_nccl(ret, "Recv");
    }

    pub fn group_start(&self) {
        let ret = unsafe { ncclGroupStart() };
        check_nccl(ret, "GroupStart");
    }

    pub fn group_end(&self) {
        let ret = unsafe { ncclGroupEnd() };
        check_nccl(ret, "GroupEnd");
    }
}

impl Drop for NcclPipeline {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe { ncclCommDestroy(self.comm) };
        }
    }
}

unsafe impl Send for NcclPipeline {}
