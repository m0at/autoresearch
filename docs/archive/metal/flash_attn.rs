use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use half::bf16;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

const METAL_SOURCE: &str = include_str!("../metal/attention.metal");

static ATTN_PIPELINE_CACHE: OnceLock<Mutex<HashMap<String, ComputePipeline>>> = OnceLock::new();

fn get_attn_pipeline(
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &str,
) -> Result<ComputePipeline> {
    let cache = ATTN_PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("lock: {e}")))?;
    if let Some(p) = map.get(kernel_name) {
        return Ok(p.clone());
    }
    let library = device
        .new_library_with_source(METAL_SOURCE, None)
        .map_err(|e| candle_core::Error::Msg(format!("compile attention.metal: {e}")))?;
    let function = library
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("get_function({kernel_name}): {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(format!("pipeline({kernel_name}): {e}")))?;
    map.insert(kernel_name.to_string(), pipeline.clone());
    Ok(pipeline)
}

fn extract_f32(s: &CpuStorage, l: &Layout) -> Result<Vec<f32>> {
    let (a, b) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("non-contiguous".into()))?;
    match s {
        CpuStorage::F32(d) => Ok(d[a..b].to_vec()),
        CpuStorage::BF16(d) => Ok(d[a..b].iter().map(|v| v.to_f32()).collect()),
        _ => Err(candle_core::Error::Msg("unsupported dtype".into())),
    }
}

fn to_cpu(data: Vec<f32>, dtype: DType) -> CpuStorage {
    match dtype {
        DType::BF16 => CpuStorage::BF16(data.into_iter().map(bf16::from_f32).collect()),
        _ => CpuStorage::F32(data),
    }
}

fn sdtype(s: &CpuStorage) -> DType {
    match s {
        CpuStorage::BF16(_) => DType::BF16,
        _ => DType::F32,
    }
}

fn t2f(t: &Tensor) -> Result<Vec<f32>> {
    let (g, l) = t.storage_and_layout();
    match &*g {
        Storage::Cpu(s) => extract_f32(s, l),
        _ => Err(candle_core::Error::Msg("expected cpu".into())),
    }
}

// ── CPU attention ───────────────────────────────────────────────────────────

fn cpu_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    d: usize,
    ws: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let hpk = nh / nkv;
    let stride_b = t * nh * d;
    let stride_bk = t * nkv * d;
    // Parallel over (batch, head) pairs -- each writes to a non-overlapping output region
    let chunks: Vec<(usize, usize, Vec<f32>)> = (0..b * nh)
        .into_par_iter()
        .map(|bh| {
            let bi = bh / nh;
            let hi = bh % nh;
            let kh = hi / hpk;
            let mut chunk = vec![0f32; t * d];
            for qi in 0..t {
                let qo = bi * stride_b + qi * nh * d + hi * d;
                let mut mx = f32::NEG_INFINITY;
                let mut sc = Vec::with_capacity(t);
                for ki in 0..t {
                    if ki > qi || (ws > 0 && qi - ki >= ws) {
                        sc.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let ko = bi * stride_bk + ki * nkv * d + kh * d;
                    let mut dot = 0f32;
                    for dd in 0..d {
                        dot += q[qo + dd] * k[ko + dd];
                    }
                    dot *= scale;
                    mx = mx.max(dot);
                    sc.push(dot);
                }
                let mut se = 0f32;
                let es: Vec<f32> = sc
                    .iter()
                    .map(|&s| {
                        let e = (s - mx).exp();
                        se += e;
                        e
                    })
                    .collect();
                let co = qi * d;
                for dd in 0..d {
                    let mut a = 0f32;
                    for ki in 0..t {
                        a += es[ki] * v[bi * stride_bk + ki * nkv * d + kh * d + dd];
                    }
                    chunk[co + dd] = if se > 0.0 { a / se } else { 0.0 };
                }
            }
            (bi, hi, chunk)
        })
        .collect();
    // Scatter chunks into output
    let mut out = vec![0f32; b * t * nh * d];
    for (bi, hi, chunk) in chunks {
        for qi in 0..t {
            let dst = bi * stride_b + qi * nh * d + hi * d;
            let src = qi * d;
            out[dst..dst + d].copy_from_slice(&chunk[src..src + d]);
        }
    }
    out
}

fn softmax_row(
    q: &[f32],
    k: &[f32],
    qo: usize,
    kh: usize,
    bi: usize,
    qi: usize,
    t: usize,
    nkv: usize,
    d: usize,
    ws: usize,
    scale: f32,
) -> Vec<f32> {
    let mut sc = vec![f32::NEG_INFINITY; t];
    let mut mx = f32::NEG_INFINITY;
    for ki in 0..t {
        if ki > qi || (ws > 0 && qi - ki >= ws) {
            continue;
        }
        let ko = bi * t * nkv * d + ki * nkv * d + kh * d;
        let mut dot = 0f32;
        for dd in 0..d {
            dot += q[qo + dd] * k[ko + dd];
        }
        dot *= scale;
        sc[ki] = dot;
        mx = mx.max(dot);
    }
    let mut se = 0f32;
    let mut p = vec![0f32; t];
    for ki in 0..t {
        let e = (sc[ki] - mx).exp();
        p[ki] = e;
        se += e;
    }
    if se > 0.0 {
        for ki in 0..t {
            p[ki] /= se;
        }
    }
    p
}

fn cpu_bwd_dq(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &[f32],
    g: &[f32],
    b: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    d: usize,
    ws: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let hpk = nh / nkv;
    let stride_b = t * nh * d;
    let stride_bk = t * nkv * d;
    // Parallel over (batch, head) -- each writes to non-overlapping dq region
    let chunks: Vec<(usize, usize, Vec<f32>)> = (0..b * nh)
        .into_par_iter()
        .map(|bh| {
            let bi = bh / nh;
            let hi = bh % nh;
            let kh = hi / hpk;
            let mut chunk = vec![0f32; t * d];
            for qi in 0..t {
                let qo = bi * stride_b + qi * nh * d + hi * d;
                let p = softmax_row(q, k, qo, kh, bi, qi, t, nkv, d, ws, scale);
                let mut di = 0f32;
                for dd in 0..d {
                    di += g[qo + dd] * o[qo + dd];
                }
                let co = qi * d;
                for ki in 0..t {
                    if p[ki] == 0.0 {
                        continue;
                    }
                    let ko = bi * stride_bk + ki * nkv * d + kh * d;
                    let mut dp = 0f32;
                    for dd in 0..d {
                        dp += g[qo + dd] * v[ko + dd];
                    }
                    let ds = p[ki] * (dp - di);
                    for dd in 0..d {
                        chunk[co + dd] += ds * k[ko + dd] * scale;
                    }
                }
            }
            (bi, hi, chunk)
        })
        .collect();
    let mut dq = vec![0f32; b * t * nh * d];
    for (bi, hi, chunk) in chunks {
        for qi in 0..t {
            let dst = bi * stride_b + qi * nh * d + hi * d;
            let src = qi * d;
            dq[dst..dst + d].copy_from_slice(&chunk[src..src + d]);
        }
    }
    dq
}

fn cpu_bwd_dkv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &[f32],
    g: &[f32],
    b: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    d: usize,
    ws: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (d as f32).sqrt();
    let hpk = nh / nkv;
    let stride_b = t * nh * d;
    let stride_bk = t * nkv * d;
    // Parallel over batch -- each batch element has independent dk/dv
    let batch_results: Vec<(Vec<f32>, Vec<f32>)> = (0..b)
        .into_par_iter()
        .map(|bi| {
            let mut dk_local = vec![0f32; t * nkv * d];
            let mut dv_local = vec![0f32; t * nkv * d];
            for kh in 0..nkv {
                for qh in 0..hpk {
                    let hi = kh * hpk + qh;
                    for qi in 0..t {
                        let qo = bi * stride_b + qi * nh * d + hi * d;
                        let p = softmax_row(q, k, qo, kh, bi, qi, t, nkv, d, ws, scale);
                        let mut di = 0f32;
                        for dd in 0..d {
                            di += g[qo + dd] * o[qo + dd];
                        }
                        for ki in 0..t {
                            if p[ki] == 0.0 {
                                continue;
                            }
                            let ko_global = bi * stride_bk + ki * nkv * d + kh * d;
                            let ko_local = ki * nkv * d + kh * d;
                            for dd in 0..d {
                                dv_local[ko_local + dd] += p[ki] * g[qo + dd];
                            }
                            let mut dp = 0f32;
                            for dd in 0..d {
                                dp += g[qo + dd] * v[ko_global + dd];
                            }
                            let ds = p[ki] * (dp - di);
                            for dd in 0..d {
                                dk_local[ko_local + dd] += ds * q[qo + dd] * scale;
                            }
                        }
                    }
                }
            }
            (dk_local, dv_local)
        })
        .collect();
    // Combine per-batch results into global arrays
    let mut dk = vec![0f32; b * t * nkv * d];
    let mut dv = vec![0f32; b * t * nkv * d];
    for (bi, (dk_local, dv_local)) in batch_results.into_iter().enumerate() {
        let off = bi * stride_bk;
        dk[off..off + stride_bk].copy_from_slice(&dk_local);
        dv[off..off + stride_bk].copy_from_slice(&dv_local);
    }
    (dk, dv)
}

// ── FlashAttentionOp (CustomOp3) ────────────────────────────────────────────

struct FlashAttentionOp {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    window_size: usize,
}

impl CustomOp3 for FlashAttentionOp {
    fn name(&self) -> &'static str {
        "flash_attention_forward"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dt = sdtype(s1);
        let q = extract_f32(s1, l1)?;
        let k = extract_f32(s2, l2)?;
        let v = extract_f32(s3, l3)?;
        let d = l1.shape().dims();
        let (b, t) = (d[0], d[1]);
        let out = cpu_fwd(
            &q,
            &k,
            &v,
            b,
            t,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            self.window_size,
        );
        Ok((
            to_cpu(out, dt),
            Shape::from_dims(&[b, t, self.n_head, self.head_dim]),
        ))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let dev = s1.device();
        let dims = l1.shape().dims();
        let (b, t) = (dims[0], dims[1]);
        let os = Shape::from_dims(&[b, t, self.n_head, self.head_dim]);
        let ob = dev.new_buffer(os.elem_count(), s1.dtype(), "fa_out")?;
        let p = get_attn_pipeline(dev.device(), "flash_attention_forward")?;
        let e = dev.command_encoder()?;
        e.set_compute_pipeline_state(&p);
        e.set_buffer(
            0,
            Some(s1.buffer()),
            l1.start_offset() * s1.dtype().size_in_bytes(),
        );
        e.set_buffer(
            1,
            Some(s2.buffer()),
            l2.start_offset() * s2.dtype().size_in_bytes(),
        );
        e.set_buffer(
            2,
            Some(s3.buffer()),
            l3.start_offset() * s3.dtype().size_in_bytes(),
        );
        e.set_buffer(3, Some(&ob), 0);
        e.set_bytes(4, &(b as u32));
        e.set_bytes(5, &(t as u32));
        e.set_bytes(6, &(self.n_head as u32));
        e.set_bytes(7, &(self.n_kv_head as u32));
        e.set_bytes(8, &(self.head_dim as u32));
        e.set_bytes(9, &(self.window_size as u32));
        // Br = 16 in the Metal kernel
        e.dispatch_thread_groups(
            MTLSize {
                width: (t + 15) / 16,
                height: self.n_head,
                depth: b,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        e.use_resource(s1.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(s2.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(s3.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(&*ob, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(ob, dev.clone(), os.elem_count(), s1.dtype()),
            os,
        ))
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let q = if q.is_contiguous() {
            q.clone()
        } else {
            q.contiguous()?
        };
        let k = if k.is_contiguous() {
            k.clone()
        } else {
            k.contiguous()?
        };
        let v = if v.is_contiguous() {
            v.clone()
        } else {
            v.contiguous()?
        };
        let o = if res.is_contiguous() {
            res.clone()
        } else {
            res.contiguous()?
        };
        let go = if grad.is_contiguous() {
            grad.clone()
        } else {
            grad.contiguous()?
        };
        let (nh, nkv, hd, ws) = (self.n_head, self.n_kv_head, self.head_dim, self.window_size);

        let dq = q.apply_op2_no_bwd(
            &go,
            &BwdDqOp {
                nh,
                nkv,
                hd,
                ws,
                k: k.clone(),
                v: v.clone(),
                o: o.clone(),
            },
        )?;
        let dkv = k.apply_op2_no_bwd(
            &v,
            &BwdDkvOp {
                nh,
                nkv,
                hd,
                ws,
                q,
                o,
                go,
            },
        )?;
        let dk = dkv.narrow(0, 0, 1)?.squeeze(0)?;
        let dv = dkv.narrow(0, 1, 1)?.squeeze(0)?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

// ── BwdDqOp ─────────────────────────────────────────────────────────────────

struct BwdDqOp {
    nh: usize,
    nkv: usize,
    hd: usize,
    ws: usize,
    k: Tensor,
    v: Tensor,
    o: Tensor,
}

impl CustomOp2 for BwdDqOp {
    fn name(&self) -> &'static str {
        "flash_attention_bwd_dq"
    }

    fn cpu_fwd(
        &self,
        sq: &CpuStorage,
        lq: &Layout,
        sg: &CpuStorage,
        lg: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dt = sdtype(sq);
        let q = extract_f32(sq, lq)?;
        let go = extract_f32(sg, lg)?;
        let k = t2f(&self.k)?;
        let v = t2f(&self.v)?;
        let o = t2f(&self.o)?;
        let s = lq.shape().clone();
        let d = s.dims();
        let (b, t) = (d[0], d[1]);
        let dq = cpu_bwd_dq(
            &q, &k, &v, &o, &go, b, t, self.nh, self.nkv, self.hd, self.ws,
        );
        Ok((to_cpu(dq, dt), s))
    }

    fn metal_fwd(
        &self,
        sq: &MetalStorage,
        lq: &Layout,
        sg: &MetalStorage,
        lg: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let dev = sq.device();
        let s = lq.shape().clone();
        let dims = s.dims();
        let (b, t) = (dims[0], dims[1]);
        let buf = dev.new_buffer(s.elem_count(), sq.dtype(), "fa_dq")?;

        let kg = self.k.storage_and_layout();
        let vg = self.v.storage_and_layout();
        let og = self.o.storage_and_layout();
        let (km, kl) = match &*kg.0 {
            Storage::Metal(m) => (m, kg.1),
            _ => return Err(candle_core::Error::Msg("expected metal".into())),
        };
        let (vm, vl) = match &*vg.0 {
            Storage::Metal(m) => (m, vg.1),
            _ => return Err(candle_core::Error::Msg("expected metal".into())),
        };
        let (om, ol) = match &*og.0 {
            Storage::Metal(m) => (m, og.1),
            _ => return Err(candle_core::Error::Msg("expected metal".into())),
        };

        let p = get_attn_pipeline(dev.device(), "flash_attention_bwd_dq")?;
        let e = dev.command_encoder()?;
        e.set_compute_pipeline_state(&p);
        e.set_buffer(
            0,
            Some(sq.buffer()),
            lq.start_offset() * sq.dtype().size_in_bytes(),
        );
        e.set_buffer(
            1,
            Some(km.buffer()),
            kl.start_offset() * km.dtype().size_in_bytes(),
        );
        e.set_buffer(
            2,
            Some(vm.buffer()),
            vl.start_offset() * vm.dtype().size_in_bytes(),
        );
        e.set_buffer(
            3,
            Some(om.buffer()),
            ol.start_offset() * om.dtype().size_in_bytes(),
        );
        e.set_buffer(
            4,
            Some(sg.buffer()),
            lg.start_offset() * sg.dtype().size_in_bytes(),
        );
        e.set_buffer(5, Some(&buf), 0);
        e.set_bytes(6, &(b as u32));
        e.set_bytes(7, &(t as u32));
        e.set_bytes(8, &(self.nh as u32));
        e.set_bytes(9, &(self.nkv as u32));
        e.set_bytes(10, &(self.hd as u32));
        e.set_bytes(11, &(self.ws as u32));
        // Br_bwd = 16 in the Metal kernel
        e.dispatch_thread_groups(
            MTLSize {
                width: (t + 15) / 16,
                height: self.nh,
                depth: b,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        e.use_resource(sq.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(km.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(vm.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(om.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(sg.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(&*buf, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(buf, dev.clone(), s.elem_count(), sq.dtype()),
            s,
        ))
    }
}

// ── BwdDkvOp ────────────────────────────────────────────────────────────────

struct BwdDkvOp {
    nh: usize,
    nkv: usize,
    hd: usize,
    ws: usize,
    q: Tensor,
    o: Tensor,
    go: Tensor,
}

impl CustomOp2 for BwdDkvOp {
    fn name(&self) -> &'static str {
        "flash_attention_bwd_dkv"
    }

    fn cpu_fwd(
        &self,
        sk: &CpuStorage,
        lk: &Layout,
        sv: &CpuStorage,
        lv: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dt = sdtype(sk);
        let k = extract_f32(sk, lk)?;
        let v = extract_f32(sv, lv)?;
        let q = t2f(&self.q)?;
        let o = t2f(&self.o)?;
        let go = t2f(&self.go)?;
        let dims = lk.shape().dims();
        let (b, t) = (dims[0], dims[1]);
        let (dk, dv) = cpu_bwd_dkv(
            &q, &k, &v, &o, &go, b, t, self.nh, self.nkv, self.hd, self.ws,
        );
        let mut out = Vec::with_capacity(dk.len() + dv.len());
        out.extend_from_slice(&dk);
        out.extend_from_slice(&dv);
        Ok((
            to_cpu(out, dt),
            Shape::from_dims(&[2, b, t, self.nkv, self.hd]),
        ))
    }

    fn metal_fwd(
        &self,
        sk: &MetalStorage,
        lk: &Layout,
        sv: &MetalStorage,
        lv: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let dev = sk.device();
        let dims = lk.shape().dims();
        let (b, t) = (dims[0], dims[1]);
        let ne = b * t * self.nkv * self.hd;
        let es = sk.dtype().size_in_bytes();
        let cb = dev.new_buffer(ne * 2, sk.dtype(), "fa_dkv")?;

        let qg = self.q.storage_and_layout();
        let og = self.o.storage_and_layout();
        let gg = self.go.storage_and_layout();
        let (qm, ql) = match &*qg.0 {
            Storage::Metal(m) => (m, qg.1),
            _ => return Err(candle_core::Error::Msg("expected metal".into())),
        };
        let (om, ol) = match &*og.0 {
            Storage::Metal(m) => (m, og.1),
            _ => return Err(candle_core::Error::Msg("expected metal".into())),
        };
        let (gm, gl) = match &*gg.0 {
            Storage::Metal(m) => (m, gg.1),
            _ => return Err(candle_core::Error::Msg("expected metal".into())),
        };

        let p = get_attn_pipeline(dev.device(), "flash_attention_bwd_dkv")?;
        let e = dev.command_encoder()?;
        e.set_compute_pipeline_state(&p);
        e.set_buffer(
            0,
            Some(qm.buffer()),
            ql.start_offset() * qm.dtype().size_in_bytes(),
        );
        e.set_buffer(
            1,
            Some(sk.buffer()),
            lk.start_offset() * sk.dtype().size_in_bytes(),
        );
        e.set_buffer(
            2,
            Some(sv.buffer()),
            lv.start_offset() * sv.dtype().size_in_bytes(),
        );
        e.set_buffer(
            3,
            Some(om.buffer()),
            ol.start_offset() * om.dtype().size_in_bytes(),
        );
        e.set_buffer(
            4,
            Some(gm.buffer()),
            gl.start_offset() * gm.dtype().size_in_bytes(),
        );
        e.set_buffer(5, Some(&cb), 0);
        e.set_buffer(6, Some(&cb), ne * es);
        e.set_bytes(7, &(b as u32));
        e.set_bytes(8, &(t as u32));
        e.set_bytes(9, &(self.nh as u32));
        e.set_bytes(10, &(self.nkv as u32));
        e.set_bytes(11, &(self.hd as u32));
        e.set_bytes(12, &(self.ws as u32));
        // Bc_bwd = 8 in the Metal kernel — grid over KV blocks
        e.dispatch_thread_groups(
            MTLSize {
                width: (t + 7) / 8,
                height: self.nkv,
                depth: b,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        e.use_resource(qm.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(sk.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(sv.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(om.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(gm.buffer(), objc2_metal::MTLResourceUsage::Read);
        e.use_resource(&*cb, objc2_metal::MTLResourceUsage::Write);
        Ok((
            MetalStorage::new(cb, dev.clone(), ne * 2, sk.dtype()),
            Shape::from_dims(&[2, b, t, self.nkv, self.hd]),
        ))
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn fused_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    window_size: usize,
) -> Result<Tensor> {
    let q = if q.is_contiguous() {
        q.clone()
    } else {
        q.contiguous()?
    };
    let k = if k.is_contiguous() {
        k.clone()
    } else {
        k.contiguous()?
    };
    let v = if v.is_contiguous() {
        v.clone()
    } else {
        v.contiguous()?
    };
    q.apply_op3(
        &k,
        &v,
        FlashAttentionOp {
            n_head,
            n_kv_head,
            head_dim,
            window_size,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_flash_attn_cpu_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, nkv, d) = (1, 4, 2, 2, 8);
        let q = Tensor::randn(0f32, 0.5, (b, t, nh, d), dev)?;
        let k = Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?;
        let v = Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?;
        let out = fused_flash_attention(&q, &k, &v, nh, nkv, d, 0)?;
        assert_eq!(out.dims(), &[b, t, nh, d]);
        let data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for &x in &data {
            assert!(x.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_gqa() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, nkv, d) = (1, 4, 4, 2, 8);
        let out = fused_flash_attention(
            &Tensor::randn(0f32, 0.5, (b, t, nh, d), dev)?,
            &Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?,
            &Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?,
            nh,
            nkv,
            d,
            0,
        )?;
        assert_eq!(out.dims(), &[b, t, nh, d]);
        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_grad() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, nkv, d) = (1, 4, 2, 2, 8);
        let q = candle_core::Var::from_tensor(&Tensor::randn(0f32, 0.5, (b, t, nh, d), dev)?)?;
        let k = candle_core::Var::from_tensor(&Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?)?;
        let v = candle_core::Var::from_tensor(&Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?)?;
        let out =
            fused_flash_attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), nh, nkv, d, 0)?;
        let grads = out.sum_all()?.backward()?;
        let dq = grads.get(q.as_tensor()).expect("dq");
        let dk = grads.get(k.as_tensor()).expect("dk");
        let dv = grads.get(v.as_tensor()).expect("dv");
        assert_eq!(dq.dims(), &[b, t, nh, d]);
        assert_eq!(dk.dims(), &[b, t, nkv, d]);
        assert_eq!(dv.dims(), &[b, t, nkv, d]);
        for (n, g) in [("dq", dq), ("dk", dk), ("dv", dv)] {
            let data: Vec<f32> = g.flatten_all()?.to_vec1()?;
            for &x in &data {
                assert!(x.is_finite(), "{n} not finite: {x}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_grad_gqa() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, t, nh, nkv, d) = (1, 4, 4, 2, 8);
        let q = candle_core::Var::from_tensor(&Tensor::randn(0f32, 0.5, (b, t, nh, d), dev)?)?;
        let k = candle_core::Var::from_tensor(&Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?)?;
        let v = candle_core::Var::from_tensor(&Tensor::randn(0f32, 0.5, (b, t, nkv, d), dev)?)?;
        let out =
            fused_flash_attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), nh, nkv, d, 0)?;
        let grads = out.sum_all()?.backward()?;
        assert_eq!(grads.get(q.as_tensor()).unwrap().dims(), &[b, t, nh, d]);
        assert_eq!(grads.get(k.as_tensor()).unwrap().dims(), &[b, t, nkv, d]);
        assert_eq!(grads.get(v.as_tensor()).unwrap().dims(), &[b, t, nkv, d]);
        Ok(())
    }
}

// ── Metal GPU gradient tests ────────────────────────────────────────────────
#[cfg(test)]
#[cfg(feature = "metal")]
mod metal_tests {
    use super::*;
    use candle_core::{DType, Device};

    // BF16 tolerance: forward is tight (tiling matches CPU), backward accumulates
    // more error due to recomputed softmax in bf16 and GPU thread ordering.
    const ATOL: f64 = 0.025;
    const RTOL: f64 = 0.025;
    // Allow up to 0.5% of elements to exceed tolerance (GPU non-determinism)
    const MAX_OUTLIER_FRAC: f64 = 0.005;

    fn metal_device() -> Device {
        Device::new_metal(0).expect("Metal device required")
    }

    fn pseudo_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i32) as f32 / i32::MAX as f32 * scale
            })
            .collect()
    }

    fn to_f32_cpu(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn assert_close(got: &[f32], expected: &[f32], label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut worst_i = 0;
        let mut fail_count = 0usize;
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let ae = (g as f64 - e as f64).abs();
            let re = if e.abs() > 1e-6 {
                ae / e.abs() as f64
            } else {
                ae
            };
            if ae > max_abs {
                max_abs = ae;
                worst_i = i;
            }
            if re > max_rel {
                max_rel = re;
            }
            if ae >= ATOL && re >= RTOL {
                fail_count += 1;
                if fail_count <= 5 {
                    eprintln!(
                        "  FAIL {label}[{i}]: got {g}, expected {e}, abs={ae:.6}, rel={re:.6}"
                    );
                }
            }
        }
        eprintln!(
            "  {label}: max_abs={max_abs:.6}@[{worst_i}], max_rel={max_rel:.6}, fails={fail_count}/{}",
            got.len()
        );
        let max_outliers = (got.len() as f64 * MAX_OUTLIER_FRAC).ceil() as usize;
        assert!(
            fail_count <= max_outliers,
            "{label}: {fail_count}/{} elements exceed tolerance (atol={ATOL}, rtol={RTOL}, max_outliers={max_outliers})",
            got.len()
        );
    }

    /// Run flash attention on Metal and CPU, compare forward + backward.
    /// Uses F32 for CPU reference (full precision) and BF16 on Metal.
    fn run_grad_check(
        b: usize,
        t: usize,
        nh: usize,
        nkv: usize,
        d: usize,
        ws: usize,
        seed: u64,
        label: &str,
    ) -> Result<()> {
        let metal = metal_device();
        let cpu = &Device::Cpu;

        // Generate deterministic data (small scale to avoid softmax saturation)
        let qd = pseudo_random(b * t * nh * d, seed, 0.3);
        let kd = pseudo_random(b * t * nkv * d, seed + 1, 0.3);
        let vd = pseudo_random(b * t * nkv * d, seed + 2, 0.3);

        // ── CPU reference (BF16 inputs like Metal, F32 internal computation) ──
        // Round-trip through BF16 so CPU inputs match Metal exactly.
        let q_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&qd[..], cpu)?
                .reshape((b, t, nh, d))?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?,
        )?;
        let k_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&kd[..], cpu)?
                .reshape((b, t, nkv, d))?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?,
        )?;
        let v_cpu = candle_core::Var::from_tensor(
            &Tensor::new(&vd[..], cpu)?
                .reshape((b, t, nkv, d))?
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?,
        )?;
        let out_cpu = fused_flash_attention(
            q_cpu.as_tensor(),
            k_cpu.as_tensor(),
            v_cpu.as_tensor(),
            nh,
            nkv,
            d,
            ws,
        )?;
        let grads_cpu = out_cpu.sum_all()?.backward()?;
        let dq_cpu = grads_cpu.get(q_cpu.as_tensor()).expect("cpu dq");
        let dk_cpu = grads_cpu.get(k_cpu.as_tensor()).expect("cpu dk");
        let dv_cpu = grads_cpu.get(v_cpu.as_tensor()).expect("cpu dv");

        // ── Metal (BF16, uses Metal kernels) ──
        let q_metal = candle_core::Var::from_tensor(
            &Tensor::new(&qd[..], cpu)?
                .reshape((b, t, nh, d))?
                .to_dtype(DType::BF16)?
                .to_device(&metal)?,
        )?;
        let k_metal = candle_core::Var::from_tensor(
            &Tensor::new(&kd[..], cpu)?
                .reshape((b, t, nkv, d))?
                .to_dtype(DType::BF16)?
                .to_device(&metal)?,
        )?;
        let v_metal = candle_core::Var::from_tensor(
            &Tensor::new(&vd[..], cpu)?
                .reshape((b, t, nkv, d))?
                .to_dtype(DType::BF16)?
                .to_device(&metal)?,
        )?;
        let out_metal = fused_flash_attention(
            q_metal.as_tensor(),
            k_metal.as_tensor(),
            v_metal.as_tensor(),
            nh,
            nkv,
            d,
            ws,
        )?;

        // Compare forward outputs
        let out_cpu_vals = to_f32_cpu(&out_cpu);
        let out_metal_vals = to_f32_cpu(&out_metal);
        assert_close(&out_metal_vals, &out_cpu_vals, &format!("{label}_fwd"));

        // Backward
        let grads_metal = out_metal.to_dtype(DType::F32)?.sum_all()?.backward()?;
        let dq_metal = grads_metal.get(q_metal.as_tensor()).expect("metal dq");
        let dk_metal = grads_metal.get(k_metal.as_tensor()).expect("metal dk");
        let dv_metal = grads_metal.get(v_metal.as_tensor()).expect("metal dv");

        // Compare gradients
        assert_close(
            &to_f32_cpu(dq_metal),
            &to_f32_cpu(dq_cpu),
            &format!("{label}_dq"),
        );
        assert_close(
            &to_f32_cpu(dk_metal),
            &to_f32_cpu(dk_cpu),
            &format!("{label}_dk"),
        );
        assert_close(
            &to_f32_cpu(dv_metal),
            &to_f32_cpu(dv_cpu),
            &format!("{label}_dv"),
        );

        Ok(())
    }

    // ── Forward-only sanity check ──────────────────────────────────────────

    #[test]
    fn metal_flash_attn_fwd_sanity() -> Result<()> {
        // Verify causal masking: Metal forward matches CPU forward
        let metal = metal_device();
        let cpu = &Device::Cpu;
        let (b, t, nh, nkv, d) = (1, 4, 1, 1, 64);
        let n = b * t * nh * d;
        let qd = pseudo_random(n, 42, 0.3);
        let kd = pseudo_random(n, 43, 0.3);
        let vd = pseudo_random(n, 44, 0.3);

        let q_metal = Tensor::new(&qd[..], cpu)?
            .reshape((b, t, nh, d))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let k_metal = Tensor::new(&kd[..], cpu)?
            .reshape((b, t, nkv, d))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let v_metal = Tensor::new(&vd[..], cpu)?
            .reshape((b, t, nkv, d))?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let out_metal = fused_flash_attention(&q_metal, &k_metal, &v_metal, nh, nkv, d, 0)?;

        let q_cpu = Tensor::new(&qd[..], cpu)?
            .reshape((b, t, nh, d))?
            .to_dtype(DType::BF16)?;
        let k_cpu = Tensor::new(&kd[..], cpu)?
            .reshape((b, t, nkv, d))?
            .to_dtype(DType::BF16)?;
        let v_cpu = Tensor::new(&vd[..], cpu)?
            .reshape((b, t, nkv, d))?
            .to_dtype(DType::BF16)?;
        let out_cpu = fused_flash_attention(&q_cpu, &k_cpu, &v_cpu, nh, nkv, d, 0)?;

        assert_close(&to_f32_cpu(&out_metal), &to_f32_cpu(&out_cpu), "fwd_sanity");
        Ok(())
    }

    // ── Basic MHA (n_head == n_kv_head) ─────────────────────────────────────

    #[test]
    fn metal_flash_attn_grad_basic() -> Result<()> {
        // B=2, T=32, n_head=2, n_kv_head=2, head_dim=64
        run_grad_check(2, 32, 2, 2, 64, 0, 42, "basic_b2t32")
    }

    #[test]
    fn metal_flash_attn_grad_basic_small() -> Result<()> {
        // Smaller case: B=1, T=16, to verify tiling edge cases
        run_grad_check(1, 16, 2, 2, 64, 0, 100, "basic_b1t16")
    }

    // ── GQA (n_head=4, n_kv_head=2) ────────────────────────────────────────

    #[test]
    fn metal_flash_attn_grad_gqa() -> Result<()> {
        // B=2, T=32, n_head=4, n_kv_head=2, head_dim=64
        run_grad_check(2, 32, 4, 2, 64, 0, 77, "gqa_b2t32")
    }

    #[test]
    fn metal_flash_attn_grad_gqa_small() -> Result<()> {
        // B=1, T=16, n_head=4, n_kv_head=2, head_dim=64
        run_grad_check(1, 16, 4, 2, 64, 0, 200, "gqa_b1t16")
    }

    // ── Sliding window (window_size = T/2) ──────────────────────────────────

    #[test]
    fn metal_flash_attn_grad_window() -> Result<()> {
        // B=2, T=32, n_head=2, n_kv_head=2, head_dim=64, window_size=16
        run_grad_check(2, 32, 2, 2, 64, 16, 55, "window_b2t32w16")
    }

    #[test]
    fn metal_flash_attn_grad_gqa_window() -> Result<()> {
        // GQA + sliding window: B=2, T=32, n_head=4, n_kv_head=2, head_dim=64, window_size=16
        run_grad_check(2, 32, 4, 2, 64, 16, 99, "gqa_window_b2t32w16")
    }

    /// Naive matmul attention on Metal (the CUDA/CPU fallback path from gpt.rs).
    /// Input: Q [B, T, n_head, head_dim], K [B, T, n_kv_head, head_dim], V same as K.
    /// Causal masking via pre-computed mask. Returns [B, T, n_head, head_dim].
    fn naive_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        causal_mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _nh, _d) = q.dims4()?;
        // Transpose to (B, n_head, T, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GQA expand
        let (k, v) = if n_kv_head < n_head {
            let rep = n_head / n_kv_head;
            let k = k
                .unsqueeze(2)?
                .expand((b, n_kv_head, rep, t, head_dim))?
                .reshape((b, n_head, t, head_dim))?;
            let v = v
                .unsqueeze(2)?
                .expand((b, n_kv_head, rep, t, head_dim))?
                .reshape((b, n_head, t, head_dim))?;
            (k, v)
        } else {
            (k, v)
        };

        let n_h = n_head;
        let q3 = q.contiguous()?.reshape((b * n_h, t, head_dim))?;
        let k3 = k.contiguous()?.reshape((b * n_h, t, head_dim))?;
        let att = (q3.matmul(&k3.transpose(1, 2)?)? * scale as f64)?;
        let att = att.reshape((b, n_h, t, t))?;

        // Apply causal mask
        let valid = causal_mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(att.dims())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, att.device())?
            .to_dtype(att.dtype())?
            .broadcast_as(att.dims())?;
        let att = valid.where_cond(&att, &neg_inf)?;

        // Softmax last dim
        let max_val = att.max_keepdim(candle_core::D::Minus1)?;
        let att = att.broadcast_sub(&max_val)?;
        let exp_att = att.exp()?;
        let sum_exp = exp_att.sum_keepdim(candle_core::D::Minus1)?;
        let att = exp_att.broadcast_div(&sum_exp)?;

        let att3 = att.reshape((b * n_h, t, t))?;
        let v3 = v.contiguous()?.reshape((b * n_h, t, head_dim))?;
        let y = att3.matmul(&v3)?;
        // (B, n_head, T, head_dim) -> (B, T, n_head, head_dim)
        y.reshape((b, n_h, t, head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Force GPU sync by pulling a scalar to CPU.
    fn metal_sync(t: &Tensor) -> Result<()> {
        let _ = t.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_flash_vs_naive_attention() -> Result<()> {
        use std::time::Instant;

        let metal = metal_device();
        let cpu = &Device::Cpu;

        // Model config: B=8, T=2048, n_head=4, n_kv_head=4, head_dim=128
        let (b, t, nh, nkv, d) = (8, 2048, 4, 4, 128);
        let iters = 10;

        eprintln!("=== Flash vs Naive Attention Benchmark ===");
        eprintln!("B={b}, T={t}, n_head={nh}, n_kv_head={nkv}, head_dim={d}");
        eprintln!("Iterations: {iters}");
        eprintln!();

        // Create random BF16 tensors on Metal
        let q = Tensor::randn(0f32, 0.3, (b, t, nh, d), cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let k = Tensor::randn(0f32, 0.3, (b, t, nkv, d), cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;
        let v = Tensor::randn(0f32, 0.3, (b, t, nkv, d), cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&metal)?;

        // Pre-compute causal mask (U8, [T, T]) on Metal
        let rows = Tensor::arange(0i64, t as i64, &metal)?
            .reshape((t, 1))?
            .broadcast_as((t, t))?;
        let cols = Tensor::arange(0i64, t as i64, &metal)?
            .reshape((1, t))?
            .broadcast_as((t, t))?;
        let causal_mask = rows.ge(&cols)?.contiguous()?;

        // Warmup: one run of each to compile pipelines and warm caches
        let out_flash = fused_flash_attention(&q, &k, &v, nh, nkv, d, 0)?;
        metal_sync(&out_flash)?;
        let out_naive = naive_attention(&q, &k, &v, nh, nkv, d, &causal_mask)?;
        metal_sync(&out_naive)?;

        // Benchmark flash attention
        let mut flash_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let start = Instant::now();
            let out = fused_flash_attention(&q, &k, &v, nh, nkv, d, 0)?;
            metal_sync(&out)?;
            let elapsed = start.elapsed();
            flash_times.push(elapsed.as_secs_f64() * 1000.0);
            eprintln!("  flash [{i}]: {:.2} ms", flash_times[i]);
        }

        // Benchmark naive attention
        let mut naive_times = Vec::with_capacity(iters);
        for i in 0..iters {
            let start = Instant::now();
            let out = naive_attention(&q, &k, &v, nh, nkv, d, &causal_mask)?;
            metal_sync(&out)?;
            let elapsed = start.elapsed();
            naive_times.push(elapsed.as_secs_f64() * 1000.0);
            eprintln!("  naive [{i}]: {:.2} ms", naive_times[i]);
        }

        // Compute stats (skip first iteration as additional warmup)
        let flash_skip = if iters > 2 { 1 } else { 0 };
        let naive_skip = if iters > 2 { 1 } else { 0 };
        let flash_avg: f64 =
            flash_times[flash_skip..].iter().sum::<f64>() / (iters - flash_skip) as f64;
        let naive_avg: f64 =
            naive_times[naive_skip..].iter().sum::<f64>() / (iters - naive_skip) as f64;
        let flash_min = flash_times[flash_skip..]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let naive_min = naive_times[naive_skip..]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let speedup_avg = naive_avg / flash_avg;
        let speedup_min = naive_min / flash_min;

        // Memory: naive materializes [B, n_head, T, T] attention matrix
        let attn_matrix_bytes = b * nh * t * t * 2; // BF16 = 2 bytes
        let attn_matrix_mb = attn_matrix_bytes as f64 / (1024.0 * 1024.0);

        eprintln!();
        eprintln!("=== Results ===");
        eprintln!("Flash attention:  avg {flash_avg:.2} ms, min {flash_min:.2} ms");
        eprintln!("Naive attention:  avg {naive_avg:.2} ms, min {naive_min:.2} ms");
        eprintln!("Speedup:          avg {speedup_avg:.2}x, min {speedup_min:.2}x");
        eprintln!(
            "Naive attn matrix: {attn_matrix_mb:.1} MB  (B={b} * nh={nh} * T={t} * T={t} * 2B)"
        );
        eprintln!();

        // Sanity: both should produce finite output
        assert!(flash_avg.is_finite());
        assert!(naive_avg.is_finite());

        Ok(())
    }
}
