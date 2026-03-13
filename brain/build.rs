fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-arch=sm_90")
        .flag("--use_fast_math")
        .flag("-O3")
        .flag("-diag-suppress=177"); // suppress unused variable warnings from headers

    for kernel in &[
        "embedding",
        "relu_sq",
        "residual_scale",
        "softcap",
        "ve_apply",
        "elementwise",
        "adamw",
        "muon",
        "rms_norm",
        "fused_norm_residual",
        "rope",
        "cross_entropy",
        "layer_stat",
    ] {
        build.file(format!("kernels/{kernel}.cu"));
    }

    build.compile("engine_kernels");

    // Link prebuilt flash-attn v3 (Hopper)
    // Set FLASH_ATTN_V3_BUILD_DIR to override. Default: fa3/build or ../prebuilt.
    let flash_dir = std::env::var("FLASH_ATTN_V3_BUILD_DIR")
        .unwrap_or_else(|_| "fa3/build".to_string());
    let flash_lib = if std::path::Path::new(&format!("{flash_dir}/libflashattention3.a")).exists() {
        "flashattention3"
    } else {
        // Fallback for builds without FA3 (will fail to link if ffi uses v3 symbols)
        eprintln!("WARNING: FA3 library not found at {flash_dir}/libflashattention3.a");
        eprintln!("         Falling back to FA2 at ../prebuilt (may cause linker errors)");
        println!("cargo:rustc-link-search=native=../prebuilt");
        "flashattention"
    };
    println!("cargo:rerun-if-changed={flash_dir}/lib{flash_lib}.a");
    println!("cargo:rustc-link-search=native={flash_dir}");
    println!("cargo:rustc-link-lib=static={flash_lib}");

    // Flash-attn depends on CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
