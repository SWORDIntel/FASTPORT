# FASTPORT DSLLVM Build Notes

This file describes how to integrate FASTPORT with your DSLLVM toolchain. Treat it as a **template** that you adapt to your exact DSLLVM installation.

---

## 1. Assumptions

- DSLLVM is installed under `/opt/dsllvm` (or similar), providing:
  - `clang`, `clang++`, `lld`, and DSLLVM-specific optimisation/hardening passes.
- Either:
  - (A) you have a Rust toolchain that uses DSLLVM as its LLVM backend, **or**
  - (B) you will selectively compile C/C++ hot-path components with DSLLVM and call them from Rust via FFI.

Update paths and commands below to match your environment.

---

## 2. Rust-Only DSLLVM Path (If Available)

If you maintain a Rust toolchain backed by DSLLVM (custom `rustc`):

1. Export the DSLLVM-backed `rustc`:

   ```bash
   export RUSTC=/opt/dsllvm/bin/rustc
   ```

2. Add DSLLVM-specific flags via `RUSTFLAGS`:

   ```bash
   export RUSTFLAGS="-C target-cpu=meteorlake -C target-feature=+avx512f,+avx512bw -C lto=yes"
   ```

3. In `.cargo/config.toml`, define explicit build profiles:

   ```toml
   [build]
   rustc = "/opt/dsllvm/bin/rustc"

   [profile.release-fastport-fast]
   opt-level = "z"
   lto = true
   codegen-units = 1

   [profile.release-fastport-secure]
   opt-level = "s"
   lto = true
   codegen-units = 1
   panic = "abort"
   ```

4. Build with the desired profile and features (AVX-512 / AVX2).

---

## 3. Hybrid C/C++ + Rust Path (Recommended Starting Point)

If you prefer keeping upstream Rust stable and using DSLLVM for C/C++ hot paths:

1. Identify the core packets/IO module where most CPU time is spent.
2. Re-implement that module in C or C++ with DSLLVM-specific tuning.
3. Compile with DSLLVM, for example:

   ```bash
   /opt/dsllvm/bin/clang -O3 -march=meteorlake -flto      -c src/dsllvm_packet_core.c -o target/dsllvm_packet_core.o
   ```

4. Expose a small, stable C API:

   ```c
   // dsllvm_packet_core.h
   #pragma once
   #include <stdint.h>

   typedef struct {
       const char *ip;
       uint16_t port;
       uint8_t protocol; // 0=tcp,1=udp,2=quic
   } fp_target_t;

   void fp_scan_batch(const fp_target_t *targets, uint32_t len);
   ```

5. Call this C API from Rust via FFI, keeping Rust responsible for:
   - Target enumeration and batching.
   - Integrating scan results into `threat_event_v1` events.
   - Handling tasking and telemetry.

---

## 4. Build Modes

Add simple wrappers in `build.sh` or `Cargo.toml` for:

- `fastport-fast`
  - Uses DSLLVM-optimised C core and baseline Rust release build.
- `fastport-secure`
  - Enables DSLLVM hardening flags and any additional runtime checks.
- `fastport-debug`
  - Uses standard Rust toolchain with debug symbols and sanitizers.

Ensure each mode embeds its build ID and mode into FASTPORT’s telemetry output so HDAIS/ARTICBASTION can track which binaries are in use.

---

## 5. Validation

After wiring DSLLVM into FASTPORT, validate:

- No change in scan correctness on known lab ranges.  
- Measurable throughput improvement vs non-DSLLVM builds (for `fastport-fast`).  
- No unexpected crashes or instability under heavy load.  

Record benchmark results in the FASTPORT repo (e.g. `BENCHMARKS_DSLLVM.md`) for future regression testing.
