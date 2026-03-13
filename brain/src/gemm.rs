use std::sync::Arc;

use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::{self, cublasOperation_t};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;

const CUBLAS_WORKSPACE_BYTES: usize = 32 * 1024 * 1024; // 32 MB

/// Thin wrapper around cuBLAS for the matmuls we need.
///
/// All GEMMs are BF16 inputs with FP32 accumulation (CUBLAS_COMPUTE_32F).
/// cuBLAS uses column-major layout. We store row-major, so we exploit the
/// identity: row-major C = A @ B is equivalent to column-major C^T = B^T @ A^T.
pub struct GemmRunner {
    blas: CudaBlas,
    _workspace: CudaSlice<u8>,
}

impl GemmRunner {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let blas = CudaBlas::new(stream.clone()).expect("failed to create cuBLAS handle");

        // Allocate 32 MB workspace so cuBLAS can use optimal algorithms.
        let workspace: CudaSlice<u8> = stream
            .alloc_zeros::<u8>(CUBLAS_WORKSPACE_BYTES)
            .expect("failed to allocate cuBLAS workspace");
        let ws_ptr = {
            let (ptr, _sync) = workspace.device_ptr(workspace.stream());
            ptr
        };
        unsafe {
            sys::cublasSetWorkspace_v2(
                *blas.handle(),
                ws_ptr as *mut core::ffi::c_void,
                CUBLAS_WORKSPACE_BYTES,
            )
            .result()
            .expect("cublasSetWorkspace_v2 failed");
        }

        // BF16 GEMMs: use default math (tensor cores) and disallow reduced-precision
        // reductions. Without the DISALLOW flag, cuBLAS may internally downcast
        // intermediate accumulations, causing accuracy loss and slower algorithm
        // selection. PyTorch sets this flag for all BF16 matmuls.
        unsafe {
            let math_mode = sys::cublasMath_t::CUBLAS_DEFAULT_MATH as u32
                | sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION as u32;
            sys::cublasSetMathMode(*blas.handle(), std::mem::transmute(math_mode))
                .result()
                .expect("cublasSetMathMode failed");
        }

        Self {
            blas,
            _workspace: workspace,
        }
    }

    /// Access the underlying cuBLAS handle for custom GEMM configurations.
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// Y = X @ W^T
    ///
    /// Row-major: X is (M, K), W is (N, K), Y is (M, N).
    /// Column-major equivalent: Y^T(N,M) = W(N,K) @ X^T(K,M)
    ///   A = W, B = X, C = Y
    ///   transa = N (W is already (N,K) in col-major = (K,N) row-major, but we want N rows in output)
    ///   transb = T (X stored row-major (M,K) => col-major (K,M), transpose to get (M,K)^T...
    ///
    /// Actually, let's be precise about the column-major trick:
    /// Row-major: Y(M,N) = X(M,K) @ W^T(K,N)
    /// In memory, row-major Y(M,N) = col-major Y^T(N,M)
    /// So: Y^T(N,M) = (X @ W^T)^T = W @ X^T
    /// cuBLAS: C(N,M) = A(N,K) * B(K,M) where A=W, B=X^T
    /// But X is stored row-major (M,K) = col-major (K,M), which IS X^T in col-major.
    /// And W is stored row-major (N,K) = col-major (K,N), so we need transa=T to get W(N,K).
    ///
    /// Result:
    ///   transa = T, transb = N
    ///   A = W (row-major (N,K)), lda = K
    ///   B = X (row-major (M,K)), ldb = K
    ///   C = Y (row-major (M,N)), ldc = N
    ///   m_cublas = N, n_cublas = M, k_cublas = K
    pub fn matmul(
        &self,
        x: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert!(x.len() >= m * k, "x too small: {} < {}", x.len(), m * k);
        assert!(w.len() >= n * k, "w too small: {} < {}", w.len(), n * k);
        assert!(y.len() >= m * n, "y too small: {} < {}", y.len(), m * n);

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: bf16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            beta: bf16::from_f32(0.0),
            ldc: n as i32,
        };
        unsafe { self.blas.gemm(cfg, w, x, y) }.expect("cuBLAS GEMM failed");
    }

    /// dW += dY^T @ X  (accumulate weight gradients with beta=1.0)
    ///
    /// Row-major: dW(N,K) += dY^T(N,M) @ X(M,K) = dY(M,N)^T @ X(M,K)
    /// Col-major: dW^T(K,N) += X^T(K,M) @ dY(M,N)... but dW stored row-major (N,K) = col-major (K,N).
    ///
    /// We want C(K,N) += B^T(K,M) @ A(M,N)  -- no, let's redo:
    /// Row-major: dW(N,K) += dY^T @ X, where dY is (M,N), X is (M,K)
    /// In col-major terms: C^T = dW^T(K,N). We accumulate into dW(N,K).
    /// C^T(K,N) = (dY^T @ X)^T = X^T @ dY
    /// cuBLAS: C(K,N) = A(K,M) * B(M,N) with A=X^T, B=dY
    /// X stored row-major (M,K) = col-major (K,M) = X^T already. So transa=N.
    /// dY stored row-major (M,N) = col-major (N,M). We need (M,N), so transb=T.
    ///
    /// Result:
    ///   transa = N, transb = T
    ///   A = X (row-major (M,K)), lda = K
    ///   B = dY (row-major (M,N)), ldb = N
    ///   C = dW (row-major (N,K)), ldc = K
    ///   m_cublas = K, n_cublas = N, k_cublas = M
    ///   alpha = 1.0, beta = 1.0 (accumulate)
    pub fn matmul_acc(
        &self,
        dy: &CudaSlice<bf16>,
        x: &CudaSlice<bf16>,
        dw: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert!(dy.len() >= m * n, "dy too small: {} < {}", dy.len(), m * n);
        assert!(x.len() >= m * k, "x too small: {} < {}", x.len(), m * k);
        assert!(dw.len() >= n * k, "dw too small: {} < {}", dw.len(), n * k);

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: k as i32,
            n: n as i32,
            k: m as i32,
            alpha: bf16::from_f32(1.0),
            lda: k as i32,
            ldb: n as i32,
            beta: bf16::from_f32(1.0),
            ldc: k as i32,
        };
        unsafe { self.blas.gemm(cfg, x, dy, dw) }.expect("cuBLAS GEMM (acc) failed");
    }

    /// dX = dY @ W  (backward input gradient)
    ///
    /// Row-major: dX(M,K) = dY(M,N) @ W(N,K)
    /// Col-major: dX^T(K,M) = W^T(K,N) @ dY^T(N,M)
    /// cuBLAS: C(K,M) = A(K,N) * B(N,M) where A=W^T, B=dY^T
    /// W stored row-major (N,K) = col-major (K,N) = W^T already. So transa=N.
    /// dY stored row-major (M,N) = col-major (N,M) = dY^T already. So transb=N.
    ///
    /// Result:
    ///   transa = N, transb = N
    ///   A = W (row-major (N,K)), lda = K
    ///   B = dY (row-major (M,N)), ldb = N
    ///   C = dX (row-major (M,K)), ldc = K
    ///   m_cublas = K, n_cublas = M, k_cublas = N
    pub fn matmul_bwd_x(
        &self,
        dy: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        dx: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert!(dy.len() >= m * n, "dy too small: {} < {}", dy.len(), m * n);
        assert!(w.len() >= n * k, "w too small: {} < {}", w.len(), n * k);
        assert!(dx.len() >= m * k, "dx too small: {} < {}", dx.len(), m * k);

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: k as i32,
            n: m as i32,
            k: n as i32,
            alpha: bf16::from_f32(1.0),
            lda: k as i32,
            ldb: n as i32,
            beta: bf16::from_f32(0.0),
            ldc: k as i32,
        };
        unsafe { self.blas.gemm(cfg, w, dy, dx) }.expect("cuBLAS GEMM (bwd_x) failed");
    }

    // ── Batched GEMM methods for Muon optimizer ─────────────────────────────
    //
    // All batched methods operate on contiguously stacked matrices with
    // uniform stride between batches. This lets us collapse 32+ individual
    // GEMMs into a single cublasSgemmStridedBatched call.

    /// Batched Y = X @ W^T
    ///
    /// Same semantics as `matmul` but for `batch_count` stacked matrices.
    /// Each matrix X[i] is at offset i*stride_x, W[i] at i*stride_w, Y[i] at i*stride_y.
    pub fn batched_matmul(
        &self,
        x: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
    ) {
        let stride_x = (m * k) as i64;
        let stride_w = (n * k) as i64;
        let stride_y = (m * n) as i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: k as i32,
                beta: bf16::from_f32(0.0),
                ldc: n as i32,
            },
            batch_size: batch_count as i32,
            stride_a: stride_w,
            stride_b: stride_x,
            stride_c: stride_y,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, w, x, y) }
            .expect("cuBLAS batched GEMM failed");
    }

    /// Batched dW += dY^T @ X  (accumulate, beta=1.0)
    ///
    /// Same semantics as `matmul_acc` but batched.
    pub fn batched_matmul_acc(
        &self,
        dy: &CudaSlice<bf16>,
        x: &CudaSlice<bf16>,
        dw: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
    ) {
        let stride_dy = (m * n) as i64;
        let stride_x = (m * k) as i64;
        let stride_dw = (n * k) as i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: k as i32,
                n: n as i32,
                k: m as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: n as i32,
                beta: bf16::from_f32(1.0),
                ldc: k as i32,
            },
            batch_size: batch_count as i32,
            stride_a: stride_x,
            stride_b: stride_dy,
            stride_c: stride_dw,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, x, dy, dw) }
            .expect("cuBLAS batched GEMM (acc) failed");
    }

    // ── Shared-input / shared-output batched GEMM methods for packed QKV ──

    /// Y = X @ W^T, batched with shared X (stride_B = 0).
    /// Used for QKV: 3 projections in one cuBLAS call, xn read once.
    /// X: [M, K] shared input; W: [batch*N, K] stacked weights (stride_W = N*K);
    /// Y: [batch*M, N] stacked outputs (stride_Y = M*N).
    pub fn matmul_shared_x_batched(
        &self,
        x: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>,
        m: usize, n: usize, k: usize,
        batch_count: usize,
    ) {
        let stride_w = (n * k) as i64;
        let stride_x = 0i64;
        let stride_y = (m * n) as i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: k as i32,
                beta: bf16::from_f32(0.0),
                ldc: n as i32,
            },
            batch_size: batch_count as i32,
            stride_a: stride_w,
            stride_b: stride_x,
            stride_c: stride_y,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, w, x, y) }
            .expect("cuBLAS shared-X batched GEMM (QKV fwd) failed");
    }

    /// Batched dW += dY^T @ X with shared X (stride_B=0).
    /// Used for QKV dW: d_wqkv += d_qkv^T @ saved_xn.
    /// dY stacked [batch*M, N], X shared [M, K] (stride=0), dW stacked [batch*N, K].
    pub fn matmul_shared_x_batched_acc(
        &self,
        dy: &CudaSlice<bf16>,
        x: &CudaSlice<bf16>,
        dw: &mut CudaSlice<bf16>,
        m: usize, n: usize, k: usize,
        batch_count: usize,
    ) {
        let stride_dy = (m * n) as i64;
        let stride_x  = 0i64;
        let stride_dw = (n * k) as i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: k as i32,
                n: n as i32,
                k: m as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: n as i32,
                beta: bf16::from_f32(1.0),
                ldc: k as i32,
            },
            batch_size: batch_count as i32,
            stride_a: stride_x,
            stride_b: stride_dy,
            stride_c: stride_dw,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, x, dy, dw) }
            .expect("cuBLAS shared-X batched GEMM acc (QKV dW) failed");
    }

    /// Batched dX = sum_i(dY_i @ W_i), shared output with beta=1 accumulate.
    /// Used for QKV dX: d_xn = d_q @ wq + d_k @ wk + d_v @ wv.
    /// Caller must zero dx first; all batches accumulate into the same output.
    pub fn matmul_batched_bwd_x_shared_out(
        &self,
        dy: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        dx: &mut CudaSlice<bf16>,
        m: usize, n: usize, k: usize,
        batch_count: usize,
    ) {
        let stride_dy = (m * n) as i64;
        let stride_w  = (n * k) as i64;
        let stride_dx = 0i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k as i32,
                n: m as i32,
                k: n as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: n as i32,
                beta: bf16::from_f32(1.0),
                ldc: k as i32,
            },
            batch_size: batch_count as i32,
            stride_a: stride_w,
            stride_b: stride_dy,
            stride_c: stride_dx,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, w, dy, dx) }
            .expect("cuBLAS batched bwd_x shared-out (QKV dX) failed");
    }

    /// Batched dX = dY @ W  (backward input gradient)
    ///
    /// Same semantics as `matmul_bwd_x` but batched.
    pub fn batched_matmul_bwd_x(
        &self,
        dy: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        dx: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
    ) {
        let stride_dy = (m * n) as i64;
        let stride_w = (n * k) as i64;
        let stride_dx = (m * k) as i64;

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k as i32,
                n: m as i32,
                k: n as i32,
                alpha: bf16::from_f32(1.0),
                lda: k as i32,
                ldb: n as i32,
                beta: bf16::from_f32(0.0),
                ldc: k as i32,
            },
            batch_size: batch_count as i32,
            stride_a: stride_w,
            stride_b: stride_dy,
            stride_c: stride_dx,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, w, dy, dx) }
            .expect("cuBLAS batched GEMM (bwd_x) failed");
    }
}
