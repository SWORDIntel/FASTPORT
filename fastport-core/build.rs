use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("cpu_features.rs");
    let mut f = File::create(&dest_path).unwrap();

    // Detect target CPU features at compile time
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    writeln!(f, "// Auto-generated CPU feature detection").unwrap();
    writeln!(f, "// Target architecture: {}", target_arch).unwrap();
    writeln!(f, "").unwrap();

    if target_arch == "x86_64" {
        // Check for AVX-512 support
        let has_avx512 = cfg!(feature = "avx512") ||
                         target_features.contains("avx512f");

        // Check for AVX2 support
        let has_avx2 = cfg!(feature = "avx2") ||
                       target_features.contains("avx2") ||
                       has_avx512;  // AVX-512 implies AVX2

        writeln!(f, "pub const AVX512_ENABLED: bool = {};", has_avx512).unwrap();
        writeln!(f, "pub const AVX2_ENABLED: bool = {};", has_avx2).unwrap();
        writeln!(f, "").unwrap();

        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn get_simd_variant() -> &'static str {{").unwrap();
        if has_avx512 {
            writeln!(f, "    \"AVX-512\"").unwrap();
        } else if has_avx2 {
            writeln!(f, "    \"AVX2\"").unwrap();
        } else {
            writeln!(f, "    \"scalar\"").unwrap();
        }
        writeln!(f, "}}").unwrap();
        writeln!(f, "").unwrap();

        // Performance warning if not AVX-512
        if !has_avx512 {
            println!("cargo:warning=Building without AVX-512 support. Performance will be reduced.");
            println!("cargo:warning=To enable AVX-512: RUSTFLAGS='-C target-cpu=native' cargo build --release --features avx512");
        }

        if has_avx512 {
            println!("cargo:warning=AVX-512 enabled - Maximum performance mode");
        } else if has_avx2 {
            println!("cargo:warning=AVX2 fallback mode - Good performance but not optimal");
        }

    } else if target_arch == "aarch64" {
        writeln!(f, "pub const NEON_ENABLED: bool = true;").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn get_simd_variant() -> &'static str {{ \"NEON\" }}").unwrap();
    } else {
        writeln!(f, "pub const AVX512_ENABLED: bool = false;").unwrap();
        writeln!(f, "pub const AVX2_ENABLED: bool = false;").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn get_simd_variant() -> &'static str {{ \"scalar\" }}").unwrap();
    }

    // Generate P-core detection code
    writeln!(f, "").unwrap();
    writeln!(f, "/// Get number of performance cores for pinning").unwrap();
    writeln!(f, "#[inline]").unwrap();
    writeln!(f, "pub fn get_pcore_count() -> usize {{").unwrap();
    writeln!(f, "    // On hybrid architectures, assume first 50% are P-cores").unwrap();
    writeln!(f, "    let total = num_cpus::get();").unwrap();
    writeln!(f, "    std::cmp::max(total / 2, 1)").unwrap();
    writeln!(f, "}}").unwrap();

    println!("cargo:rustc-link-lib=static=c");
}
