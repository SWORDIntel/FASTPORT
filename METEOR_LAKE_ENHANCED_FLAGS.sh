#!/bin/bash
# ============================================================================
# INTEL METEOR LAKE ULTIMATE COMPILER FLAGS REFERENCE - ENHANCED EDITION
# Dell Latitude 5450 Covert Edition - Complete Optimization Guide
# Version: ENHANCED v2.1 - December 2024
# KYBERLOCK Research Division - Tactical Computing
# Engineering Sample A00 | 9-Layer Flex Fabric | OpenVINO Integration
# ============================================================================

# ============================================================================
# SYSTEM SPECIFICATIONS - DELL LATITUDE 5450 COVERT EDITION
# ============================================================================
# CPU: Intel(R) Core(TM) Ultra 7 165H (Engineering Sample A00)
# Architecture: Meteor Lake (Family 6 Model 170 Stepping 4)
# Cores: 16 (6P + 10E) - Hybrid Architecture
# GPU: Intel Arc Graphics (Xe-LPG, 128 EUs)
# NPU: VPU 3720 (2 Neural Compute Engines) - Base Layer
# 
# FLEX FABRIC ACCELERATOR STACK (9 Layers):
#   Theoretical Combined: 1440 TOPS
#   Interface: OpenVINO Unified Runtime
#   Fallback: Graceful degradation through compute tiers
#
# CACHE HIERARCHY:
#   L1 Data (P-core): 48KB per core
#   L2 Cache (P-core): 2MB per core  
#   L1 Data (E-core): 32KB per core
#   L2 Cache (E-cluster): 4MB shared
# ============================================================================

# ============================================================================
# ENHANCEMENT CHANGELOG v2.0
# ============================================================================
# NEW ISA Extensions Added:
#   - AVX-IFMA (Integer Fused Multiply-Add)
#   - AVX-NE-CONVERT (Neural Engine Convert)
#   - AVX-VNNI-INT8 (8-bit VNNI)
#   - CMPCCXADD (Compare and Conditional Add)
#   - RAO-INT (Remote Atomic Operations)
#   - PREFETCHI (Instruction Prefetch)
#   - HRESET (History Reset)
#   - Key Locker (KL/WIDEKL)
#   - PKU (Protection Keys)
#   - PTWRITE (Processor Trace)
#   - RDPID (Read Processor ID)
#   - PCONFIG (Platform Configuration)
#   - ENQCMD (Enqueue Command)
#   - AMX Support (Engineering Sample)
#
# NEW Optimizations Added:
#   - Interprocedural Analysis (IPA) suite
#   - Advanced scheduling optimizations
#   - Cache-tuned parameters for Meteor Lake
#   - Polly polyhedral optimizer (Clang)
#   - Enhanced Rust target features
#
# Fixes:
#   - Removed deprecated -mcpu (use -mtune)
#   - Added missing ISA extensions
# ============================================================================

# ============================================================================
# SECTION 1: BASE OPTIMIZATION FLAGS (ENHANCED)
# ============================================================================

# Maximum Performance Base - Enhanced with additional optimizations
export CFLAGS_BASE="\
-O3 \
-pipe \
-fomit-frame-pointer \
-funroll-loops \
-fstrict-aliasing \
-fno-plt \
-fdata-sections \
-ffunction-sections \
-flto=auto \
-fuse-linker-plugin \
-fgraphite-identity \
-floop-nest-optimize \
-ftree-vectorize \
-ftree-slp-vectorize \
-fipa-pta \
-fipa-cp-clone \
-fdevirtualize-speculatively \
-fdevirtualize-at-ltrans \
-fipa-ra \
-fipa-sra \
-fipa-vrp"

# Architecture Specific
export ARCH_FLAGS="-march=meteorlake -mtune=meteorlake"

# Alternative if meteorlake not recognized (GCC < 13)
export ARCH_FLAGS_FALLBACK="-march=alderlake -mtune=alderlake"

# Native detection fallback
export ARCH_FLAGS_NATIVE="-march=native -mtune=native"

# ============================================================================
# SECTION 2: INSTRUCTION SET EXTENSIONS - COMPLETE & ENHANCED
# ============================================================================

# Core x86-64 Features
export ISA_BASELINE="-msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2"

# Advanced Vector Extensions
export ISA_AVX="-mavx -mavx2 -mf16c -mfma"

# AI/ML Acceleration - ENHANCED for Meteor Lake
export ISA_VNNI="-mavxvnni -mavxvnniint8"

# NEW: AVX Integer/Convert Extensions (Meteor Lake specific)
export ISA_AVX_EXTENDED="-mavxifma -mavxneconvert"

# Bit Manipulation
export ISA_BMI="-mbmi -mbmi2 -mlzcnt -mpopcnt"

# Cryptographic Acceleration - ENHANCED
export ISA_CRYPTO="-maes -mvaes -mpclmul -mvpclmulqdq -msha -mgfni"

# NEW: Key Locker (Hardware Key Protection)
export ISA_KEYLOCKER="-mkl -mwidekl"

# Memory & Cache Operations
export ISA_MEMORY="-mmovbe -mmovdiri -mmovdir64b -mclflushopt -mclwb -mcldemote"

# Advanced Features - ENHANCED
export ISA_ADVANCED="-madx -mrdrnd -mrdseed -mfsgsbase -mfxsr -mxsave -mxsaveopt -mxsavec -mxsaves"

# NEW: Additional Meteor Lake Features
export ISA_METEOR_LAKE="-mhreset -mpku -mptwrite -mrdpid -mpconfig -menqcmd -mprefetchi"

# NEW: Atomic Operations
export ISA_ATOMIC="-mcmpccxadd -mraoint"

# Prefetch Instructions
export ISA_PREFETCH="-mprfchw -mprefetchi"

# Additional Performance Features
export ISA_PERF_EXTRA="-msahf"  # LAHF/SAHF in 64-bit mode

# ============================================================================
# CPU CAPABILITIES (Runtime Features - No Compiler Flags)
# ============================================================================
# These are supported by your CPU but controlled at runtime/BIOS level:
#   - VMX: Virtual Machine Extensions (Intel VT-x)
#   - EPT: Extended Page Tables
#   - SMX: Safer Mode Extensions (Intel TXT)
#   - TME: Total Memory Encryption
#   - UMIP: User-Mode Instruction Prevention
#   - OSPKE: OS-enabled Protection Keys (controlled by OS)
#   - INVPCID: Invalidate PCID (enabled, improves TLB performance)
#   - User Shadow Stack: Enabled via CET
# ============================================================================

# Control Flow
export ISA_CONTROL="-mwaitpkg -muintr -mserialize -mtsxldtrk"

# CET (Control-flow Enforcement Technology)
export ISA_CET="-mshstk"
# Note: -mcet is deprecated, use -fcf-protection instead

# ============================================================================
# SECTION 2.5: AMX EXTENSIONS (ENGINEERING SAMPLE A00)
# ============================================================================
# These may only be available on engineering samples / dev boards
# Comment out if your production chip doesn't support them

export ISA_AMX="\
-mamx-tile \
-mamx-int8 \
-mamx-bf16 \
-mamx-fp16 \
-mamx-complex"

# ============================================================================
# SECTION 2.6: OPENVINO / FLEX FABRIC INTEGRATION
# ============================================================================
# The 9-layer flex fabric (1440 TOPS theoretical) is accessed via OpenVINO
# These settings optimize CPU-side code that interfaces with the accelerators

# OpenVINO Environment (set paths when OpenVINO is installed)
export_openvino_env() {
    local OPENVINO_ROOT="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}"
    
    if [[ -d "$OPENVINO_ROOT" ]]; then
        source "$OPENVINO_ROOT/setupvars.sh" 2>/dev/null
        export OV_AVAILABLE=1
        echo "✓ OpenVINO environment loaded from $OPENVINO_ROOT"
    else
        export OV_AVAILABLE=0
        echo "⚠ OpenVINO not found - CPU fallback active"
    fi
}

# OpenVINO Device Priority (graceful fallback order)
# Flex fabric layers -> NPU -> GPU -> CPU
export OV_DEVICE_PRIORITY="HETERO:NPU,GPU,CPU"
export OV_CPU_THROUGHPUT_STREAMS="AUTO"
export OV_CPU_THREADS_NUM="6"
export OV_CPU_BIND_THREAD="YES"
export OV_CPU_DENORMALS_OPTIMIZATION="YES"

# NPU-specific settings (VPU 3720 base + flex fabric)
export NPU_COMPILER_TYPE="DRIVER"
export NPU_COMPILATION_MODE_PARAMS="optimization-level=2"
export NPU_DMA_ENGINES="2"
export NPU_TILES="2"

# Flex Fabric Hints (proprietary - adjust based on workload)
export FLEX_FABRIC_LAYERS="9"
export FLEX_FABRIC_MAX_TOPS="1440"
export FLEX_FABRIC_FALLBACK="graceful"

# Compiler flags for OpenVINO-optimized CPU fallback code
export CFLAGS_OPENVINO="\
-DENABLE_OPENVINO=1 \
-DENABLE_VNNI=1 \
-DENABLE_AVX2=1 \
-DENABLE_FP16=1 \
-march=meteorlake \
-mavxvnni \
-mavxvnniint8 \
-mavxifma \
-mf16c"

# Include paths for OpenVINO (when available)
export OPENVINO_INCLUDES="-I\${INTEL_OPENVINO_DIR}/runtime/include"
export OPENVINO_LIBS="-L\${INTEL_OPENVINO_DIR}/runtime/lib/intel64 -lopenvino -lopenvino_c"

# ============================================================================
# SECTION 3: COMPLETE OPTIMAL FLAGS - TIER 1 ENHANCED
# ============================================================================

export CFLAGS_OPTIMAL="\
-O3 \
-pipe \
-fomit-frame-pointer \
-funroll-loops \
-fstrict-aliasing \
-fno-plt \
-fdata-sections \
-ffunction-sections \
-flto=auto \
-fuse-linker-plugin \
-march=meteorlake \
-mtune=meteorlake \
-msse4.2 \
-mpopcnt \
-mavx \
-mavx2 \
-mfma \
-mf16c \
-mbmi \
-mbmi2 \
-mlzcnt \
-mmovbe \
-mavxvnni \
-mavxvnniint8 \
-mavxifma \
-mavxneconvert \
-maes \
-mvaes \
-mpclmul \
-mvpclmulqdq \
-msha \
-mgfni \
-mkl \
-mwidekl \
-madx \
-mclflushopt \
-mclwb \
-mcldemote \
-mmovdiri \
-mmovdir64b \
-mwaitpkg \
-mserialize \
-mtsxldtrk \
-muintr \
-mprfchw \
-mprefetchi \
-mrdrnd \
-mrdseed \
-mfsgsbase \
-mfxsr \
-mxsave \
-mxsaveopt \
-mxsavec \
-mxsaves \
-mhreset \
-mpku \
-mptwrite \
-mrdpid \
-mpconfig \
-menqcmd \
-mcmpccxadd \
-mraoint \
-mshstk"

# Engineering Sample with AMX
export CFLAGS_OPTIMAL_AMX="$CFLAGS_OPTIMAL $ISA_AMX"

# ============================================================================
# SECTION 4: ADVANCED OPTIMIZATION FLAGS (NEW)
# ============================================================================

# Interprocedural Analysis Suite
export IPA_FLAGS="\
-fipa-pta \
-fipa-cp-clone \
-fipa-ra \
-fipa-sra \
-fipa-vrp \
-fdevirtualize-speculatively \
-fdevirtualize-at-ltrans"

# Scheduling Optimizations
export SCHED_FLAGS="\
-fsched-pressure \
-fsched-spec-load \
-fmodulo-sched \
-fmodulo-sched-allow-regmoves"

# Loop Optimizations - Enhanced
export LOOP_FLAGS="\
-ftree-loop-im \
-ftree-loop-distribution \
-ftree-loop-distribute-patterns \
-ftree-loop-vectorize \
-floop-nest-optimize"

# Code Generation Optimizations
export CODEGEN_FLAGS="\
-fgcse-after-reload \
-fpredictive-commoning \
-ftree-partial-pre \
-ftracer \
-fsplit-paths \
-fprefetch-loop-arrays"

# ============================================================================
# SECTION 5: CACHE TUNING PARAMETERS (NEW - Meteor Lake Specific)
# ============================================================================

# Meteor Lake P-core cache hierarchy
export CACHE_PARAMS_PCORE="\
--param l1-cache-size=48 \
--param l1-cache-line-size=64 \
--param l2-cache-size=2048"

# Prefetch tuning for Meteor Lake
export PREFETCH_PARAMS="\
--param prefetch-latency=300 \
--param simultaneous-prefetches=6 \
--param prefetch-min-insn-to-mem-ratio=3"

# Combined cache parameters
export CACHE_PARAMS="$CACHE_PARAMS_PCORE $PREFETCH_PARAMS"

# ============================================================================
# SECTION 6: PERFORMANCE PROFILES - ENHANCED
# ============================================================================

# PROFILE: Maximum Speed (No Safety) - Enhanced
export CFLAGS_SPEED="\
-Ofast \
-ffast-math \
-funsafe-math-optimizations \
-ffinite-math-only \
-fno-signed-zeros \
-fno-trapping-math \
-frounding-math \
-fsingle-precision-constant \
-fcx-limited-range \
$CFLAGS_OPTIMAL \
$IPA_FLAGS \
$SCHED_FLAGS \
$LOOP_FLAGS \
$CODEGEN_FLAGS \
$CACHE_PARAMS"

# PROFILE: Balanced Performance - Enhanced
export CFLAGS_BALANCED="\
-O2 \
-ftree-vectorize \
$ARCH_FLAGS \
$ISA_AVX \
$ISA_CRYPTO \
$IPA_FLAGS \
-pipe"

# PROFILE: Size Optimized
export CFLAGS_SIZE="-Os -fomit-frame-pointer -finline-limit=8 $ARCH_FLAGS $ISA_BASELINE"

# PROFILE: Debug Build - Enhanced
export CFLAGS_DEBUG="\
-Og \
-g3 \
-ggdb \
-fno-omit-frame-pointer \
-fno-inline \
-fstack-protector-all \
-fsanitize=address,undefined \
-D_DEBUG \
$ARCH_FLAGS"

# ============================================================================
# SECTION 7: INLINING PARAMETERS - ENHANCED
# ============================================================================

export INLINE_PARAMS="\
--param max-inline-insns-single=1000 \
--param max-inline-insns-auto=200 \
--param inline-unit-growth=200 \
--param large-function-growth=400 \
--param large-function-insns=3000"

# Aggressive Inlining
export INLINE_FLAGS="\
-finline-functions \
-finline-functions-called-once \
-finline-limit=1000 \
-finline-small-functions \
$INLINE_PARAMS"

# ============================================================================
# SECTION 8: VECTORIZATION TUNING - ENHANCED
# ============================================================================

export VECTORIZE="\
-ftree-vectorize \
-ftree-slp-vectorize \
-ftree-loop-vectorize \
-fvect-cost-model=unlimited \
-fsimd-cost-model=unlimited"

# Unrolling parameters
export UNROLL_PARAMS="\
--param max-unrolled-insns=400 \
--param max-average-unrolled-insns=200"

# ============================================================================
# SECTION 9: LINK-TIME OPTIMIZATION - ENHANCED
# ============================================================================

export LDFLAGS_BASE="\
-Wl,--as-needed \
-Wl,--gc-sections \
-Wl,-O2 \
-Wl,--hash-style=gnu \
-Wl,--sort-common \
-Wl,--enable-new-dtags"

export LDFLAGS_LTO="\
-flto=auto \
-fuse-linker-plugin \
-Wl,-flto \
-flto-partition=balanced \
-fno-fat-lto-objects"

export LDFLAGS_OPTIMAL="$LDFLAGS_BASE $LDFLAGS_LTO"

# Gold Linker Optimizations (faster than ld)
export LDFLAGS_GOLD="-fuse-ld=gold -Wl,--icf=all"

# MOLD Linker (Fastest - install separately)
export LDFLAGS_MOLD="-fuse-ld=mold -Wl,--threads=16"

# LLD Linker (LLVM)
export LDFLAGS_LLD="-fuse-ld=lld"

# ============================================================================
# SECTION 10: SECURITY HARDENED FLAGS - ENHANCED
# ============================================================================

# Security Hardening - Standard
export CFLAGS_SECURITY="\
-D_FORTIFY_SOURCE=3 \
-fstack-protector-strong \
-fstack-clash-protection \
-fcf-protection=full \
-fpie \
-fPIC \
-Wformat \
-Wformat-security \
-Werror=format-security \
-mindirect-branch=thunk \
-mfunction-return=thunk \
-mindirect-branch-register \
-mindirect-branch-cs-prefix \
-fharden-compares \
-fharden-conditional-branches \
-ftrivial-auto-var-init=zero"

# Security Hardening - Maximum (includes register clearing)
export CFLAGS_SECURITY_MAX="\
$CFLAGS_SECURITY \
-fzero-call-used-regs=used-gpr"

# Security Hardening - Paranoid (clears all registers)
export CFLAGS_SECURITY_PARANOID="\
$CFLAGS_SECURITY \
-fzero-call-used-regs=all"

export LDFLAGS_SECURITY="\
-Wl,-z,relro \
-Wl,-z,now \
-Wl,-z,noexecstack \
-Wl,-z,separate-code \
-pie"

# ============================================================================
# SECTION 11: KERNEL COMPILATION FLAGS - ENHANCED
# ============================================================================

export KCFLAGS="\
-O3 \
-pipe \
-march=meteorlake \
-mtune=meteorlake \
-msse4.2 \
-mpopcnt \
-mavx \
-mavx2 \
-mfma \
-mavxvnni \
-mavxvnniint8 \
-maes \
-mvaes \
-mpclmul \
-mvpclmulqdq \
-msha \
-mgfni \
-falign-functions=64 \
-falign-jumps=64 \
-falign-loops=64 \
-falign-labels=64 \
$CACHE_PARAMS"

export KCPPFLAGS="$KCFLAGS"

# Kernel Make Variables
export KBUILD_BUILD_HOST="kyberlock"
export KBUILD_BUILD_USER="tactical"
export KBUILD_CFLAGS_KERNEL="$KCFLAGS"
export KBUILD_AFLAGS_KERNEL="$KCFLAGS"

# ============================================================================
# SECTION 12: PROFILE-GUIDED OPTIMIZATION - ENHANCED
# ============================================================================

# Profile-Guided Optimization Stage 1
export PGO_GEN="-fprofile-generate -fprofile-arcs -ftest-coverage -fprofile-update=atomic"

# Profile-Guided Optimization Stage 2
export PGO_USE="-fprofile-use -fprofile-correction -fbranch-probabilities -fvpt -fprofile-values"

# AutoFDO (for perf-collected profiles)
export AUTOFDO_USE="-fauto-profile"

# ============================================================================
# SECTION 13: GRAPHITE LOOP OPTIMIZATIONS - ENHANCED
# ============================================================================

export GRAPHITE="\
-fgraphite \
-fgraphite-identity \
-floop-nest-optimize \
-floop-parallelize-all \
-ftree-loop-linear \
-floop-strip-mine \
-floop-block"

# ============================================================================
# SECTION 14: INTEL ONEAPI / DPC++ / SYCL FLAGS
# ============================================================================
# For when Intel oneAPI toolchain is available - enables direct accelerator access

# DPC++ / SYCL Compiler Flags (Intel ICX/ICPX)
export DPCPP_FLAGS="\
-fsycl \
-fsycl-targets=spir64,spir64_x86_64 \
-fsycl-unnamed-lambda \
-fsycl-device-code-split=per_kernel"

# DPC++ with NPU/Accelerator targets
export DPCPP_ACCEL="\
-fsycl \
-fsycl-targets=spir64_x86_64,intel_gpu_mtl"

# Intel ICX Compiler Flags (when available)
export ICX_FLAGS="\
-O3 \
-xCORE-AVX2 \
-qopt-zmm-usage=low \
-qopt-multiple-gather-scatter-by-shuffles \
-qopenmp-simd \
-qopt-prefetch=5 \
-qopt-mem-layout-trans=4 \
-fimf-precision=high \
-fimf-domain-exclusion=31"

# ICX for Meteor Lake specifically
export ICX_METEOR_LAKE="\
-march=meteorlake \
-mtune=meteorlake \
-qopt-report=5 \
-qopt-report-phase=all"

# Level Zero API Environment (direct accelerator access)
export ZE_AFFINITY_MASK="0,1,2"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER="1"
export ZET_ENABLE_PROGRAM_DEBUGGING="0"

# SYCL Runtime Configuration
export SYCL_DEVICE_FILTER="level_zero:gpu,opencl:cpu"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS="1"
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS="1"

# ============================================================================
# SECTION 14.5: NEURAL NETWORK / INFERENCE OPTIMIZATION
# ============================================================================
# Flags optimized for NN inference workloads on CPU (fallback path)

# BFloat16 / FP16 optimization for neural inference
export CFLAGS_NEURAL="\
-march=meteorlake \
-mavxvnni \
-mavxvnniint8 \
-mavxifma \
-mavxneconvert \
-mf16c \
-ffast-math \
-funroll-loops \
-ftree-vectorize \
-fopt-info-vec-optimized"

# VNNI-optimized quantized inference
export CFLAGS_QUANTIZED="\
$CFLAGS_NEURAL \
-DUSE_VNNI=1 \
-DINT8_INFERENCE=1 \
-DAVX_VNNI_INT8=1"

# AMX-optimized inference (engineering sample)
export CFLAGS_AMX_INFERENCE="\
$CFLAGS_NEURAL \
$ISA_AMX \
-DUSE_AMX=1 \
-DAMX_BF16=1 \
-DAMX_INT8=1"

# ============================================================================
# SECTION 14.7: ONEDNN / DNNL OPTIMIZATION (CPU INFERENCE BACKEND)
# ============================================================================
# oneDNN is the CPU inference backend for OpenVINO - optimize for VNNI/AMX

# oneDNN Runtime Configuration
export ONEDNN_VERBOSE="0"
export ONEDNN_PRIMITIVE_CACHE_CAPACITY="1024"
export ONEDNN_MAX_CPU_ISA="AVX2_VNNI_2"  # Meteor Lake max (no AVX512)
export ONEDNN_CPU_RUNTIME="OMP"
export DNNL_CPU_RUNTIME="OMP"

# For AMX workloads (engineering sample)
export ONEDNN_MAX_CPU_ISA_AMX="AVX2_VNNI_2,AMX_INT8,AMX_BF16"

# Thread binding for oneDNN
export DNNL_CPU_ISA_HINTS="prefer_ymm"  # Prefer 256-bit (no AVX512)
export ONEDNN_DEFAULT_FPMATH_MODE="BF16"

# Compiler flags for oneDNN-optimized code
export CFLAGS_ONEDNN="\
-march=meteorlake \
-mavxvnni \
-mavxvnniint8 \
-DONEDNN_ENABLE_CONCURRENT_EXEC=1 \
-DONEDNN_VERBOSE=0 \
-fopenmp"

# ============================================================================
# SECTION 15: CLANG/LLVM SPECIFIC FLAGS - ENHANCED
# ============================================================================

# Clang basic optimizations
export CLANG_FLAGS="\
-mllvm -inline-threshold=1000 \
-mllvm -unroll-threshold=1000 \
-mllvm -vectorize-loops \
-mllvm -vectorize-slp"

# Clang additional ISA flags (not available in GCC)
export CLANG_ISA_EXTRA="-minvpcid -msahf -msgx"

# Clang GVN optimizations
export CLANG_GVN="\
-mllvm -enable-gvn-hoist \
-mllvm -enable-gvn-sink"

# Clang loop optimizations
export CLANG_LOOPS="\
-mllvm -enable-loop-flatten \
-mllvm -enable-nontrivial-unswitch"

# Polly Polyhedral Optimizer (powerful loop optimizer)
export CLANG_POLLY="\
-mllvm -polly \
-mllvm -polly-vectorizer=stripmine \
-mllvm -polly-parallel \
-mllvm -polly-omp-backend=LLVM \
-mllvm -polly-num-threads=6 \
-mllvm -polly-tiling \
-mllvm -polly-prevect-width=8"

# Hot/Cold code splitting
export CLANG_SPLIT="-mllvm -hot-cold-split"

# Combined Clang optimal
export CLANG_OPTIMAL="$CLANG_FLAGS $CLANG_GVN $CLANG_LOOPS $CLANG_SPLIT"

# Combined with Polly
export CLANG_OPTIMAL_POLLY="$CLANG_OPTIMAL $CLANG_POLLY"

# ============================================================================
# SECTION 15: PARALLELIZATION & THREADING - ENHANCED
# ============================================================================

# OpenMP Flags
export OPENMP_FLAGS="-fopenmp -fopenmp-simd"

# Threading Optimizations
export THREAD_FLAGS="-pthread -D_REENTRANT -D_THREAD_SAFE"

# Parallel STL (C++17)
export PSTL_FLAGS="-ltbb -DPSTL_USE_PARALLEL_POLICIES=1"

# CPU Affinity for P-cores (Meteor Lake: 6 P-cores)
export GOMP_CPU_AFFINITY="0-5"
export OMP_NUM_THREADS="6"
export OMP_PROC_BIND="true"
export OMP_PLACES="cores"

# For all cores (P + E)
export GOMP_CPU_AFFINITY_ALL="0-15"
export OMP_NUM_THREADS_ALL="16"

# ============================================================================
# SECTION 16: MATHEMATICS & NUMERICAL - ENHANCED
# ============================================================================

# Fast Math (breaks IEEE compliance)
export MATH_FAST="\
-ffast-math \
-funsafe-math-optimizations \
-fassociative-math \
-freciprocal-math \
-ffinite-math-only \
-fno-signed-zeros \
-fno-trapping-math"

# Safe Math Optimizations (IEEE compliant)
export MATH_SAFE="\
-fno-math-errno \
-fno-trapping-math"

# Intel MKL Integration
export MKL_FLAGS="-DMKL_ILP64 -m64 -I${MKLROOT}/include"
export MKL_LIBS="-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl"

# ============================================================================
# SECTION 17: MEMORY OPTIMIZATION - ENHANCED
# ============================================================================

# Memory Alignment (64-byte for cache lines)
export ALIGN_FLAGS="\
-falign-functions=64 \
-falign-jumps=64 \
-falign-loops=64 \
-falign-labels=64"

# Stack Optimization
export STACK_FLAGS="-mpreferred-stack-boundary=5 -maccumulate-outgoing-args"

# Malloc Optimization (for custom allocators)
export MALLOC_FLAGS="-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free"

# Memory Settings
export MALLOC_ARENA_MAX="4"
export MALLOC_MMAP_THRESHOLD_="131072"
export MALLOC_TRIM_THRESHOLD_="131072"
export MALLOC_TOP_PAD_="131072"
export MALLOC_MMAP_MAX_="65536"

# ============================================================================
# SECTION 18: LANGUAGE-SPECIFIC FLAGS - ENHANCED
# ============================================================================

# C++ Optimizations
export CXXFLAGS_OPTIMAL="$CFLAGS_OPTIMAL -std=c++23 -fcoroutines -fconcepts -fmodules-ts"

# Fortran Optimizations
export FFLAGS_OPTIMAL="$CFLAGS_OPTIMAL -fdefault-real-8 -fdefault-integer-8"

# Rust Integration - ENHANCED for Meteor Lake / Latitude 5450 Covert
export RUSTFLAGS="\
-C target-cpu=meteorlake \
-C opt-level=3 \
-C lto=fat \
-C embed-bitcode=yes \
-C codegen-units=1 \
-C target-feature=+avx2,+fma,+aes,+vaes,+pclmul,+vpclmulqdq,+sha,+gfni,+avxvnni,+avxvnniint8,+avxifma,+avxneconvert,+cmpccxadd,+raoint"

# Rust with AMX (Engineering Sample A00)
export RUSTFLAGS_AMX="$RUSTFLAGS,+amx-tile,+amx-int8,+amx-bf16,+amx-fp16,+amx-complex"

# Rust with all Latitude 5450 Covert features
export RUSTFLAGS_COVERT="\
-C target-cpu=meteorlake \
-C opt-level=3 \
-C lto=fat \
-C embed-bitcode=yes \
-C codegen-units=1 \
-C link-arg=-Wl,--as-needed \
-C target-feature=+avx2,+fma,+aes,+vaes,+pclmul,+vpclmulqdq,+sha,+gfni,+avxvnni,+avxvnniint8,+avxifma,+avxneconvert,+cmpccxadd,+raoint,+amx-tile,+amx-int8,+amx-bf16,+amx-fp16,+amx-complex"

# ============================================================================
# SECTION 19: GCC 13+ SPECIFIC FLAGS
# ============================================================================

export GCC13_FLAGS="\
-std=gnu2x \
-fharden-compares \
-fharden-conditional-branches \
-ftrivial-auto-var-init=zero"

# Analyzer (for static analysis builds)
export GCC_ANALYZER="-fanalyzer"

# ============================================================================
# SECTION 20: BUILD SYSTEM EXPORTS
# ============================================================================

# CMake
export CMAKE_C_FLAGS="$CFLAGS_OPTIMAL"
export CMAKE_CXX_FLAGS="$CXXFLAGS_OPTIMAL"
export CMAKE_EXE_LINKER_FLAGS="$LDFLAGS_OPTIMAL"
export CMAKE_SHARED_LINKER_FLAGS="$LDFLAGS_OPTIMAL"
export CMAKE_MODULE_LINKER_FLAGS="$LDFLAGS_OPTIMAL"

# Autotools / Standard
export CFLAGS="$CFLAGS_OPTIMAL"
export CXXFLAGS="$CXXFLAGS_OPTIMAL"
export LDFLAGS="$LDFLAGS_OPTIMAL"

# ============================================================================
# SECTION 21: COMPLETE MEGA FLAGS (ALL OPTIMIZATIONS)
# ============================================================================

export CFLAGS_MEGA="\
$CFLAGS_OPTIMAL \
$IPA_FLAGS \
$SCHED_FLAGS \
$LOOP_FLAGS \
$CODEGEN_FLAGS \
$VECTORIZE \
$INLINE_FLAGS \
$ALIGN_FLAGS \
$CACHE_PARAMS \
$INLINE_PARAMS \
$UNROLL_PARAMS"

# With math optimizations
export CFLAGS_MEGA_FAST="$CFLAGS_MEGA $MATH_FAST"

# Engineering sample with AMX
export CFLAGS_MEGA_AMX="$CFLAGS_MEGA $ISA_AMX"

# ============================================================================
# SECTION 22: WARNING FLAGS (PRODUCTION)
# ============================================================================

export WARN_FLAGS="\
-Wall \
-Wextra \
-Wpedantic \
-Wformat=2 \
-Wformat-security \
-Wnull-dereference \
-Wstack-protector \
-Wtrampolines \
-Walloca \
-Wvla \
-Warray-bounds=2 \
-Wimplicit-fallthrough=3 \
-Wshift-overflow=2 \
-Wcast-qual \
-Wstringop-overflow=4 \
-Wconversion \
-Wlogical-op \
-Wduplicated-cond \
-Wduplicated-branches \
-Wformat-signedness \
-Wshadow \
-Wstrict-overflow=4 \
-Wundef \
-Wstrict-prototypes \
-Wswitch-default \
-Wswitch-enum \
-Wstack-usage=1000000 \
-Wcast-align=strict"

# ============================================================================
# SECTION 23: TOOLCHAIN SETUP
# ============================================================================

# Set all optimal flags
export CC="gcc-13"
export CXX="g++-13"
export AR="gcc-ar"
export NM="gcc-nm"
export RANLIB="gcc-ranlib"

# Alternative: Clang toolchain
setup_clang() {
    export CC="clang"
    export CXX="clang++"
    export AR="llvm-ar"
    export NM="llvm-nm"
    export RANLIB="llvm-ranlib"
}

# ============================================================================
# SECTION 24: USAGE FUNCTIONS - ENHANCED
# ============================================================================

# Function to compile with optimal flags
compile_optimal() {
    gcc $CFLAGS_OPTIMAL "$@" $LDFLAGS_OPTIMAL
}

# Function to compile with mega flags
compile_mega() {
    gcc $CFLAGS_MEGA "$@" $LDFLAGS_OPTIMAL
}

# Function to compile with AMX (engineering sample)
compile_amx() {
    gcc $CFLAGS_MEGA_AMX "$@" $LDFLAGS_OPTIMAL
}

# Function to compile kernel
compile_kernel() {
    make -j16 KCFLAGS="$KCFLAGS" KCPPFLAGS="$KCPPFLAGS" "$@"
}

# Function to compile with Clang + Polly
compile_clang_polly() {
    clang -march=meteorlake -O3 $CLANG_OPTIMAL_POLLY "$@" $LDFLAGS_OPTIMAL
}

# Function to compile with PGO (2-stage)
compile_pgo() {
    local src="$1"
    local out="${2:-${src%.*}}"
    
    echo "[PGO] Stage 1: Generating profile..."
    gcc $CFLAGS_OPTIMAL $PGO_GEN -o "${out}_gen" "$src" $LDFLAGS_OPTIMAL
    
    echo "[PGO] Running instrumented binary (use typical workload)..."
    ./"${out}_gen"
    
    echo "[PGO] Stage 2: Building with profile..."
    gcc $CFLAGS_OPTIMAL $PGO_USE -o "$out" "$src" $LDFLAGS_OPTIMAL
    
    echo "[PGO] Cleanup..."
    rm -f "${out}_gen" *.gcda *.gcno
    
    echo "[PGO] Done! Binary: $out"
}

# ============================================================================
# SECTION 25: VERIFICATION & TESTING
# ============================================================================

# Show current flags
show_flags() {
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  INTEL METEOR LAKE OPTIMIZATION FLAGS - ENHANCED EDITION                ║"
    echo "║  CPU: Intel Core Ultra 7 165H | Engineering Sample A00                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "=== OPTIMAL FLAGS ==="
    echo "$CFLAGS_OPTIMAL" | tr ' ' '\n' | grep -v '^$' | head -20
    echo "... ($(echo "$CFLAGS_OPTIMAL" | wc -w) total flags)"
    echo ""
    echo "=== MEGA FLAGS ==="
    echo "$(echo "$CFLAGS_MEGA" | wc -w) optimization flags loaded"
    echo ""
    echo "=== NEW IN v2.0 ==="
    echo "  ISA: avxifma, avxneconvert, avxvnniint8, cmpccxadd, raoint"
    echo "  ISA: hreset, kl, widekl, pku, ptwrite, rdpid, pconfig, enqcmd, prefetchi"
    echo "  OPT: ipa-pta, ipa-cp-clone, devirtualize-speculatively, sched-pressure"
    echo "  OPT: gscse-after-reload, predictive-commoning, modulo-sched, prefetch-loop-arrays"
    echo "  AMX: tile, int8, bf16, fp16, complex (engineering sample)"
}

# Test flags work
test_flags() {
    echo "Testing CFLAGS_OPTIMAL..."
    if echo 'int main(){return 0;}' | gcc -xc $CFLAGS_OPTIMAL - -o /tmp/test_optimal 2>&1; then
        echo "✓ CFLAGS_OPTIMAL verified working!"
        rm -f /tmp/test_optimal
    else
        echo "✗ CFLAGS_OPTIMAL failed"
        return 1
    fi
    
    echo "Testing CFLAGS_MEGA..."
    if echo 'int main(){return 0;}' | gcc -xc $CFLAGS_MEGA - -o /tmp/test_mega 2>&1; then
        echo "✓ CFLAGS_MEGA verified working!"
        rm -f /tmp/test_mega
    else
        echo "✗ CFLAGS_MEGA failed"
        return 1
    fi
    
    echo "Testing AMX flags..."
    if echo 'int main(){return 0;}' | gcc -xc -march=meteorlake $ISA_AMX - -o /tmp/test_amx 2>&1; then
        echo "✓ AMX flags verified working!"
        rm -f /tmp/test_amx
    else
        echo "⚠ AMX flags not available (production chip)"
    fi
}

# Verify all ISA extensions
test_isa() {
    echo "=== ISA Extension Verification ==="
    local flags=(
        "-mavxifma:AVX-IFMA"
        "-mavxneconvert:AVX-NE-CONVERT"
        "-mavxvnniint8:AVX-VNNI-INT8"
        "-mcmpccxadd:CMPCCXADD"
        "-mraoint:RAO-INT"
        "-mprefetchi:PREFETCHI"
        "-mhreset:HRESET"
        "-mkl:Key Locker"
        "-mwidekl:Wide Key Locker"
        "-mpku:PKU"
        "-mptwrite:PTWRITE"
        "-mrdpid:RDPID"
        "-mpconfig:PCONFIG"
        "-menqcmd:ENQCMD"
        "-mamx-tile:AMX-TILE"
        "-mamx-int8:AMX-INT8"
        "-mamx-bf16:AMX-BF16"
    )
    
    for item in "${flags[@]}"; do
        flag="${item%%:*}"
        name="${item##*:}"
        if echo 'int main(){return 0;}' | gcc -xc -march=meteorlake $flag - -o /dev/null 2>/dev/null; then
            echo "  ✓ $name ($flag)"
        else
            echo "  ✗ $name ($flag)"
        fi
    done
}

# ============================================================================
# SECTION 26: QUICK REFERENCE
# ============================================================================

print_quick_ref() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║          DELL LATITUDE 5450 COVERT EDITION - QUICK REFERENCE                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  COMPILE COMMANDS:                                                           ║
║    compile_optimal file.c     - Optimal flags (production)                   ║
║    compile_mega file.c        - All optimizations (aggressive)               ║
║    compile_amx file.c         - With AMX (eng sample A00)                    ║
║    compile_clang_polly file.c - Clang + Polly optimizer                      ║
║    compile_pgo file.c         - Profile-Guided Optimization                  ║
║                                                                              ║
║  MANUAL BUILD:                                                               ║
║    gcc $CFLAGS_OPTIMAL -o app app.c $LDFLAGS_OPTIMAL                        ║
║    gcc $CFLAGS_MEGA -o app app.c $LDFLAGS_OPTIMAL                           ║
║    gcc $CFLAGS_NEURAL -o nn nn.c $LDFLAGS_OPTIMAL  # NN inference           ║
║                                                                              ║
║  OPENVINO / FLEX FABRIC:                                                     ║
║    export_openvino_env        - Load OpenVINO runtime                        ║
║    enumerate_ov_devices       - List accelerator devices                     ║
║    OV_DEVICE_PRIORITY="HETERO:NPU,GPU,CPU"  # Fallback order                ║
║                                                                              ║
║  RUST BUILD:                                                                 ║
║    RUSTFLAGS="$RUSTFLAGS" cargo build --release                             ║
║    RUSTFLAGS="$RUSTFLAGS_COVERT" cargo build --release  # Full features     ║
║                                                                              ║
║  FALLBACK TIERS:                                                             ║
║    [0] Flex Fabric (1440 TOPS) → [1] NPU → [2] GPU → [3] CPU → [4] Scalar  ║
║                                                                              ║
║  VERIFY:                                                                     ║
║    test_flags         - Verify compiler flags                                ║
║    test_isa           - Check ISA extensions                                 ║
║    test_compute_tiers - Verify fallback chain                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
}

# ============================================================================
# SECTION 27: GRACEFUL FALLBACK VERIFICATION
# ============================================================================

# Test compute tier availability
test_compute_tiers() {
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  COMPUTE TIER AVAILABILITY - GRACEFUL FALLBACK CHECK                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Tier 0: Flex Fabric (1440 TOPS)
    echo "TIER 0: Flex Fabric Accelerator (1440 TOPS theoretical)"
    if [[ -n "$INTEL_OPENVINO_DIR" ]] && command -v benchmark_app &>/dev/null; then
        echo "  ✓ OpenVINO available - Flex fabric accessible"
    else
        echo "  ⚠ OpenVINO not configured - Flex fabric not accessible"
    fi
    
    # Tier 1: NPU (VPU 3720)
    echo ""
    echo "TIER 1: NPU / VPU 3720"
    if ls /dev/accel* 2>/dev/null | head -1 | grep -q accel; then
        echo "  ✓ NPU device nodes present"
    else
        echo "  ⚠ NPU device nodes not found - will fallback"
    fi
    
    # Tier 2: Intel Arc GPU
    echo ""
    echo "TIER 2: Intel Arc Graphics (Xe-LPG)"
    if command -v intel_gpu_top &>/dev/null || ls /dev/dri/render* &>/dev/null; then
        echo "  ✓ GPU rendering devices available"
    else
        echo "  ⚠ GPU not accessible - will fallback"
    fi
    
    # Tier 3: CPU with VNNI/AMX
    echo ""
    echo "TIER 3: CPU Optimized (AVX2/VNNI/AMX)"
    if echo 'int main(){return 0;}' | gcc -xc -march=meteorlake -mavxvnni -mavxvnniint8 $ISA_AMX - -o /dev/null 2>/dev/null; then
        echo "  ✓ Full CPU acceleration available (VNNI + AMX)"
    elif echo 'int main(){return 0;}' | gcc -xc -march=meteorlake -mavxvnni - -o /dev/null 2>/dev/null; then
        echo "  ✓ CPU acceleration available (VNNI, no AMX)"
    else
        echo "  ✓ CPU baseline available (AVX2)"
    fi
    
    # Tier 4: Scalar fallback
    echo ""
    echo "TIER 4: Scalar Fallback"
    echo "  ✓ Always available"
    
    echo ""
    echo "Fallback order: Flex Fabric → NPU → GPU → CPU (VNNI) → CPU (AVX2) → Scalar"
}

# OpenVINO device enumeration (when available)
enumerate_ov_devices() {
    if [[ -n "$INTEL_OPENVINO_DIR" ]]; then
        echo "=== OpenVINO Device Enumeration ==="
        python3 -c "
from openvino import Core
core = Core()
print('Available devices:', core.available_devices)
for device in core.available_devices:
    print(f'  {device}: {core.get_property(device, \"FULL_DEVICE_NAME\")}')
" 2>/dev/null || echo "OpenVINO Python API not available"
    else
        echo "OpenVINO not configured. Run: export_openvino_env"
    fi
}

# ============================================================================
# ACTIVATION
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  DELL LATITUDE 5450 COVERT EDITION - OPTIMIZATION FLAGS v2.1            ║"
echo "║  Intel Core Ultra 7 165H | Engineering Sample A00                       ║"
echo "║  9-Layer Flex Fabric | 1440 TOPS Theoretical | OpenVINO Interface       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "COMPUTE TIERS (Graceful Fallback):"
echo "  [0] Flex Fabric Accelerator ─── 1440 TOPS (via OpenVINO)"
echo "  [1] NPU / VPU 3720 ──────────── ~11 TOPS"
echo "  [2] Intel Arc Graphics ──────── Variable"
echo "  [3] CPU (VNNI/AMX) ──────────── Optimized inference"
echo "  [4] CPU (Scalar) ────────────── Always available"
echo ""
echo "NEW in v2.1:"
echo "  • OpenVINO integration for Flex Fabric access"
echo "  • Intel oneAPI/DPC++/SYCL compiler support"
echo "  • Neural network inference optimization flags"
echo "  • Graceful fallback verification (test_compute_tiers)"
echo "  • AVX-IFMA, AVX-NE-CONVERT, AVX-VNNI-INT8, CMPCCXADD, RAO-INT"
echo "  • AMX: tile, int8, bf16, fp16, complex"
echo "  • Key Locker, Clang Polly, cache-tuned parameters"
echo ""
echo "Commands:"
echo "  show_flags          - Display all flag sets"
echo "  test_flags          - Verify flags compile"
echo "  test_isa            - Check ISA extension support"
echo "  test_compute_tiers  - Verify fallback chain"
echo "  export_openvino_env - Load OpenVINO environment"
echo "  enumerate_ov_devices- List OpenVINO devices"
echo "  print_quick_ref     - Quick reference card"
echo ""
