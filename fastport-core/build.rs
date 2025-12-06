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

    writeln!(f, "// Auto-generated CPU feature detection with runtime fallback").unwrap();
    writeln!(f, "// Target architecture: {}", target_arch).unwrap();
    writeln!(f, "// Optimized for Intel Meteor Lake / Dell Latitude 5450 Covert Edition").unwrap();
    writeln!(f, "").unwrap();

    if target_arch == "x86_64" {
        // Check for compile-time features
        let compile_avx512 = cfg!(feature = "avx512") || target_features.contains("avx512f");
        let compile_avx2 = cfg!(feature = "avx2") || target_features.contains("avx2") || compile_avx512;
        
        // NEW: Meteor Lake specific features
        let compile_vnni = cfg!(feature = "vnni") || target_features.contains("avxvnni");
        let compile_vnni_int8 = target_features.contains("avxvnniint8");
        let compile_avx_ifma = target_features.contains("avxifma");
        let compile_amx = cfg!(feature = "amx") || target_features.contains("amx-tile");
        let compile_gfni = target_features.contains("gfni");
        let compile_vaes = target_features.contains("vaes");

        writeln!(f, "pub const COMPILE_AVX512_ENABLED: bool = {};", compile_avx512).unwrap();
        writeln!(f, "pub const COMPILE_AVX2_ENABLED: bool = {};", compile_avx2).unwrap();
        writeln!(f, "pub const COMPILE_VNNI_ENABLED: bool = {};", compile_vnni).unwrap();
        writeln!(f, "pub const COMPILE_VNNI_INT8_ENABLED: bool = {};", compile_vnni_int8).unwrap();
        writeln!(f, "pub const COMPILE_AVX_IFMA_ENABLED: bool = {};", compile_avx_ifma).unwrap();
        writeln!(f, "pub const COMPILE_AMX_ENABLED: bool = {};", compile_amx).unwrap();
        writeln!(f, "pub const COMPILE_GFNI_ENABLED: bool = {};", compile_gfni).unwrap();
        writeln!(f, "pub const COMPILE_VAES_ENABLED: bool = {};", compile_vaes).unwrap();
        writeln!(f, "").unwrap();

        // Generate runtime detection functions
        writeln!(f, "/// Runtime CPU feature detection with fallback").unwrap();
        writeln!(f, "/// Optimized for Meteor Lake hybrid architecture").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn detect_runtime_simd() -> SIMDVariant {{").unwrap();
        writeln!(f, "    #[cfg(target_arch = \"x86_64\")]").unwrap();
        writeln!(f, "    {{").unwrap();
        writeln!(f, "        // Check for Meteor Lake VNNI features first").unwrap();
        writeln!(f, "        let has_avx2 = std::arch::is_x86_feature_detected!(\"avx2\");").unwrap();
        writeln!(f, "        let has_vnni = std::arch::is_x86_feature_detected!(\"avxvnni\");").unwrap();
        writeln!(f, "        let has_fma = std::arch::is_x86_feature_detected!(\"fma\");").unwrap();
        writeln!(f, "        ").unwrap();
        writeln!(f, "        if has_avx2 && has_vnni && has_fma {{").unwrap();
        writeln!(f, "            eprintln!(\"[SIMD] AVX2 + VNNI detected (Meteor Lake optimized)\");").unwrap();
        writeln!(f, "            return SIMDVariant::AVX2_VNNI;").unwrap();
        writeln!(f, "        }} else if has_avx2 {{").unwrap();
        writeln!(f, "            eprintln!(\"[SIMD] AVX2 detected\");").unwrap();
        writeln!(f, "            return SIMDVariant::AVX2;").unwrap();
        writeln!(f, "        }}").unwrap();
        writeln!(f, "        eprintln!(\"[SIMD] No AVX2 support, falling back to scalar\");").unwrap();
        writeln!(f, "        SIMDVariant::Scalar").unwrap();
        writeln!(f, "    }}").unwrap();
        writeln!(f, "    #[cfg(not(target_arch = \"x86_64\"))]").unwrap();
        writeln!(f, "    {{").unwrap();
        writeln!(f, "        SIMDVariant::Scalar").unwrap();
        writeln!(f, "    }}").unwrap();
        writeln!(f, "}}").unwrap();
        writeln!(f, "").unwrap();

        // Add SIMD variant enum with Meteor Lake support
        writeln!(f, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
        writeln!(f, "pub enum SIMDVariant {{").unwrap();
        writeln!(f, "    AVX512,").unwrap();
        writeln!(f, "    AVX2_VNNI,  // Meteor Lake optimized path").unwrap();
        writeln!(f, "    AVX2,").unwrap();
        writeln!(f, "    Scalar,").unwrap();
        writeln!(f, "}}").unwrap();
        writeln!(f, "").unwrap();

        writeln!(f, "impl SIMDVariant {{").unwrap();
        writeln!(f, "    pub fn as_str(&self) -> &'static str {{").unwrap();
        writeln!(f, "        match self {{").unwrap();
        writeln!(f, "            SIMDVariant::AVX512 => \"AVX-512\",").unwrap();
        writeln!(f, "            SIMDVariant::AVX2_VNNI => \"AVX2+VNNI (Meteor Lake)\",").unwrap();
        writeln!(f, "            SIMDVariant::AVX2 => \"AVX2\",").unwrap();
        writeln!(f, "            SIMDVariant::Scalar => \"Scalar\",").unwrap();
        writeln!(f, "        }}").unwrap();
        writeln!(f, "    }}").unwrap();
        writeln!(f, "    ").unwrap();
        writeln!(f, "    pub fn has_vnni(&self) -> bool {{").unwrap();
        writeln!(f, "        matches!(self, SIMDVariant::AVX2_VNNI)").unwrap();
        writeln!(f, "    }}").unwrap();
        writeln!(f, "}}").unwrap();
        writeln!(f, "").unwrap();

        // Backward compatibility
        writeln!(f, "pub const AVX512_ENABLED: bool = {};", compile_avx512).unwrap();
        writeln!(f, "pub const AVX2_ENABLED: bool = {};", compile_avx2).unwrap();
        writeln!(f, "").unwrap();

        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn get_simd_variant() -> &'static str {{").unwrap();
        writeln!(f, "    detect_runtime_simd().as_str()").unwrap();
        writeln!(f, "}}").unwrap();
        writeln!(f, "").unwrap();

        // Build messages
        if compile_avx512 {
            println!("cargo:warning=AVX-512 compiled but will prefer AVX2 at runtime for stability");
        } else if compile_avx2 {
            println!("cargo:warning=AVX2 mode enabled - Optimal performance and compatibility");
        } else {
            println!("cargo:warning=Building without SIMD support. Performance will be reduced.");
            println!("cargo:warning=To enable AVX2: RUSTFLAGS='-C target-cpu=native' cargo build --release --features avx2");
        }

    } else if target_arch == "aarch64" {
        writeln!(f, "pub const NEON_ENABLED: bool = true;").unwrap();
        writeln!(f, "pub const COMPILE_AVX512_ENABLED: bool = false;").unwrap();
        writeln!(f, "pub const COMPILE_AVX2_ENABLED: bool = false;").unwrap();
        writeln!(f, "").unwrap();
        writeln!(f, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
        writeln!(f, "pub enum SIMDVariant {{ NEON, Scalar }}").unwrap();
        writeln!(f, "impl SIMDVariant {{ pub fn as_str(&self) -> &'static str {{ \"NEON\" }} }}").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn detect_runtime_simd() -> SIMDVariant {{ SIMDVariant::NEON }}").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn get_simd_variant() -> &'static str {{ \"NEON\" }}").unwrap();
    } else {
        writeln!(f, "pub const AVX512_ENABLED: bool = false;").unwrap();
        writeln!(f, "pub const AVX2_ENABLED: bool = false;").unwrap();
        writeln!(f, "pub const COMPILE_AVX512_ENABLED: bool = false;").unwrap();
        writeln!(f, "pub const COMPILE_AVX2_ENABLED: bool = false;").unwrap();
        writeln!(f, "").unwrap();
        writeln!(f, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]").unwrap();
        writeln!(f, "pub enum SIMDVariant {{ Scalar }}").unwrap();
        writeln!(f, "impl SIMDVariant {{ pub fn as_str(&self) -> &'static str {{ \"Scalar\" }} }}").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn detect_runtime_simd() -> SIMDVariant {{ SIMDVariant::Scalar }}").unwrap();
        writeln!(f, "#[inline]").unwrap();
        writeln!(f, "pub fn get_simd_variant() -> &'static str {{ \"scalar\" }}").unwrap();
    }

    // Generate P-core detection code (enhanced for Meteor Lake)
    writeln!(f, "").unwrap();
    writeln!(f, "/// Get number of performance cores for thread pinning").unwrap();
    writeln!(f, "/// Meteor Lake 165H: 6 P-cores (threads 0-5), 8 E-cores (6-13), 2 LP E-cores (14-15)").unwrap();
    writeln!(f, "#[inline]").unwrap();
    writeln!(f, "pub fn get_pcore_count() -> usize {{").unwrap();
    writeln!(f, "    let total = num_cpus::get();").unwrap();
    writeln!(f, "    let physical = num_cpus::get_physical();").unwrap();
    writeln!(f, "    ").unwrap();
    writeln!(f, "    // Detect hybrid by topology heuristic").unwrap();
    writeln!(f, "    // Hybrid CPUs: logical != physical * 2 (due to E-cores without HT)").unwrap();
    writeln!(f, "    if total >= 12 && total != physical * 2 {{").unwrap();
    writeln!(f, "        // Likely hybrid - estimate P-cores").unwrap();
    writeln!(f, "        return match total {{").unwrap();
    writeln!(f, "            16 => 6,  // Meteor Lake Ultra 7 165H").unwrap();
    writeln!(f, "            14 => 6,  // Meteor Lake Ultra 5").unwrap();
    writeln!(f, "            24 => 8,  // Alder Lake i9 / Raptor Lake").unwrap();
    writeln!(f, "            20 => 6,  // Alder Lake i7").unwrap();
    writeln!(f, "            _ => std::cmp::max(total / 3, 2),").unwrap();
    writeln!(f, "        }};").unwrap();
    writeln!(f, "    }}").unwrap();
    writeln!(f, "    // Non-hybrid: use all cores").unwrap();
    writeln!(f, "    total").unwrap();
    writeln!(f, "}}").unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "/// Check if running on hybrid architecture").unwrap();
    writeln!(f, "#[inline]").unwrap();
    writeln!(f, "pub fn is_hybrid_cpu() -> bool {{").unwrap();
    writeln!(f, "    let total = num_cpus::get();").unwrap();
    writeln!(f, "    let physical = num_cpus::get_physical();").unwrap();
    writeln!(f, "    total >= 12 && total != physical * 2").unwrap();
    writeln!(f, "}}").unwrap();
}
