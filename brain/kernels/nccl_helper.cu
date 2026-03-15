#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NCCL_CHECK(cmd) do {                          \
    ncclResult_t r = cmd;                             \
    if (r != ncclSuccess) {                           \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",    \
                __FILE__, __LINE__,                   \
                ncclGetErrorString(r));               \
        abort();                                      \
    }                                                 \
} while(0)

extern "C" {

void* nccl_init(int rank, int world_size, const char* id_file) {
    ncclUniqueId id;

    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
        FILE* f = fopen(id_file, "wb");
        if (!f) { perror("fopen id_file write"); abort(); }
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);
    } else {
        // Wait for rank 0 to write the file
        for (int i = 0; i < 300; i++) {
            FILE* f = fopen(id_file, "rb");
            if (f) {
                size_t n = fread(&id, sizeof(id), 1, f);
                fclose(f);
                if (n == 1) break;
            }
            usleep(100000); // 100ms
        }
    }

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    return (void*)comm;
}

void nccl_send_bf16(void* comm, const void* buf, int count, int dst, void* stream) {
    NCCL_CHECK(ncclSend(buf, count, ncclBfloat16, dst,
                        (ncclComm_t)comm, (cudaStream_t)stream));
}

void nccl_recv_bf16(void* comm, void* buf, int count, int src, void* stream) {
    NCCL_CHECK(ncclRecv(buf, count, ncclBfloat16, src,
                        (ncclComm_t)comm, (cudaStream_t)stream));
}

void nccl_send_f32(void* comm, const void* buf, int count, int dst, void* stream) {
    NCCL_CHECK(ncclSend(buf, count, ncclFloat32, dst,
                        (ncclComm_t)comm, (cudaStream_t)stream));
}

void nccl_recv_f32(void* comm, void* buf, int count, int src, void* stream) {
    NCCL_CHECK(ncclRecv(buf, count, ncclFloat32, src,
                        (ncclComm_t)comm, (cudaStream_t)stream));
}

void nccl_destroy(void* comm) {
    ncclCommDestroy((ncclComm_t)comm);
}

} // extern "C"
