pub const VOCAB: usize = 8192;
pub const SEQ: usize = 2048;
pub const D_MODEL: usize = 512;
pub const N_HEAD: usize = 4;
pub const N_KV_HEAD: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const MLP_DIM: usize = 2048;
pub const N_LAYER: usize = 30;
pub const VE_GATE_CH: usize = 32;
pub const SOFTCAP: f32 = 15.0;
pub const EPS: f32 = 1e-5;
pub const ROPE_BASE: f64 = 200_000.0;
pub const INIT_SCALE: f64 = 0.68;

pub const VE_LAYERS: [usize; 4] = [1, 3, 5, 7];

pub const WINDOW_SIZES: [usize; N_LAYER] = [
    256, 256, 256, 256, 2048,
    256, 256, 256, 256, 2048,
    256, 256, 256, 256, 2048,
    256, 256, 256, 256, 2048,
    256, 256, 256, 256, 2048,
    256, 256, 256, 256, 2048,
];

pub fn has_ve(layer: usize) -> bool {
    matches!(layer, 1 | 3 | 5 | 7)
}
