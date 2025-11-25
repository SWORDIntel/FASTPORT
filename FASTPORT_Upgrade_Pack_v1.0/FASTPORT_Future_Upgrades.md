# FASTPORT – Future Upgrades (Engine & Integration)

**Version:** 1.0 (Future upgrades + DSLLVM plan)  
**Status:** Draft – Implementation Ready  
**Scope:** FASTPORT as the primary high-performance active discovery engine in the DSMIL / ARTICBASTION stack.  
**Out of scope:** Web UI around results (handled by ARTICBASTION / HDAIS / other consoles).

---

## 1. Role in the Ecosystem

FASTPORT is the **active discovery and vulnerability exposure engine** that complements passive sensors (HURRICANE) and higher-level analytics (HDAIS, MEMSHADOW, SPECTRA).

- **Upstream tasking**
  - ARTICBASTION issues scan tasks (targets, scopes, profiles).
  - HURRICANE edge nodes provide current exposure, tunnel info, and reachability.
  - MEMSHADOW provides historical context (previous findings, host roles).

- **Downstream consumers**
  - HDAIS: consumes FASTPORT outputs as `threat_event_v1` `service_discovery` events.
  - ARTICBASTION TIP: stores scan campaigns, links results to campaigns/cases.
  - SPECTRA: uses FASTPORT fingerprints and CVE tags as clustering features.
  - MEMSHADOW: stores scan history for long-term reasoning and training.

**Core mission:** turn raw network reachability into **structured, CVE-linked, host-aware intelligence** that can be safely combined with other sensors.

---

## 2. Data Model & Output Normalisation

### 2.1 Canonical Output

FASTPORT should emit results in the same canonical `threat_event_v1` format used by HURRICANE/HDAIS.

For each discovered service, emit a `service_discovery` event with at least:

- `event_id`: unique ID per result.  
- `time`: timestamp (UTC).  
- `source`: `"FASTPORT"` (or `"FASTPORT_EDGE"` for edge builds).  
- **Host identity**
  - `host_id`: stable identifier (IP + optional DIO/asset ID).  
  - `dio_id`: Decimal Identity Overlay ID, if available.  
  - `asn`, `geo`, `env` (prod/lab/exercise).  
- **Service info**
  - `ip`, `port`, `protocol` (tcp/udp/quic).  
  - `service_name` (e.g. http, ssh, modbus, s7).  
  - `banner`, `version`, `tls_fingerprint` (JA3/JA4 where applicable).  
- **Vulnerability intel**
  - `cves[]` (id, cvss, exploit_maturity, known_exploit).  
  - `has_rce`: bool.  
  - `max_cvss`: number.  
- **Scan metadata**
  - `scan_id`, `scan_profile`, `simd_mode` (avx512/avx2/none).  
  - `packets_sent`, `packets_recv`, `scan_duration_ms`.  

HDAIS then enriches these with `threat_score`, `risk_bucket`, `category`, and `explanation`.

### 2.2 Result Channels

Support two main output paths:

1. **Streaming**
   - Write `service_discovery` events to a queue/stream (Redis Streams / Kafka topic `fastport.results`).  
   - ARTICBASTION + HDAIS subscribe.

2. **Batch**
   - Write NDJSON or Parquet files with normalised events for offline ingestion.  
   - Used for large campaign scans and lab datasets.

---

## 3. Engine Enhancements

### 3.1 Scan Profiles

Define named profiles tuned for specific missions:

- `recon-fast`
  - Narrow port set, aggressive timing, default TCP connect or raw packets.  
  - Use for broad initial sweeps.

- `deep-service`
  - Full port range or curated large list (e.g. top 10k).  
  - Extra banner grabs, protocol-specific handshakes.  
  - Enables extended CVE matching.

- `ics-safe`
  - ICS/SCADA ports only (Modbus, DNP3, S7, IEC-104, OPC-UA).  
  - **Read-only, safety-conscious probes** (no writes/state changes).  
  - Lower rate limits and strict time windows.

- `cloud-inventory`
  - Optimised for public cloud ranges and ephemeral hosts.  
  - Integrates with cloud metadata (tagging by provider / region).

- `dio-overlay`
  - Works over the Decimal Identity Overlay via HURRICANE tunnels.  
  - Uses DIO IDs as primary keys and logs tunnel path info for HDAIS.

Profiles are configured via YAML and selectable via CLI/TUI/API.

### 3.2 Fingerprinting & Protocol Coverage

Enhance service detection and fingerprinting:

- **TLS / HTTPS**
  - Collect JA3/JA4 TLS fingerprints.  
  - Extract certificate metadata (CN, SANs, expiry).  

- **HTTP / Web**
  - Detect framework / server (nginx, apache, IIS, gunicorn, etc.).  
  - Basic CMS/framework detection where safe.

- **Non-Web**
  - SSH version parsing and key types.  
  - Databases (MySQL/Postgres/Redis/Mongo) banner detection.  
  - Message queues and brokers (RabbitMQ, Kafka, MQTT).  

- **ICS / OT**
  - Modbus function code queries (read-only).  
  - S7, DNP3, IEC-104 handshake-level detection.  
  - Tag ICS devices with `ics_device_type` where possible.

### 3.3 Evasion & OPSEC Controls

Add optional **tradecraft-aware** features (only when permitted by ROE):  

- Rate-shaping and jitter to mimic normal patterns.  
- Randomised source ports / packet timing.  
- Application-layer handshakes tuned to look like common clients.  
- Strict legal/ROE flags to disable when not allowed.

Expose these as profile-level toggles, and record them in scan metadata.

---

## 4. CVE & Intelligence Integration

### 4.1 CVE Sources

- Local mirror or cached index of NVD (and other feeds if present).  
- Allow plug-ins for vendor advisories and curated intel feeds.  

### 4.2 CVE Matching Strategy

- Use service fingerprint + version + optional TLS cert data.  
- Prioritise:
  - RCE and auth bypass vulnerabilities.  
  - ICS/OT and SCADA-specific CVEs.  
  - High EPSS / known exploited CVEs where data is available.  

### 4.3 HDAIS Alignment

- For each discovered service, compute preliminary risk hints:  
  - `prelim_risk_score` (0–100).  
  - `prelim_risk_bucket`.  
  - `flags[]` (e.g. `["rce_detected", "ics_unsafe_port_exposed"]`).  

These are **hints**; HDAIS owns the final `threat_score` and `risk_bucket` but can reuse FASTPORT’s pre-score to save work and improve responsiveness.

---

## 5. Engine API & ARTICBASTION Integration

### 5.1 Tasking API

Expose a simple API (gRPC/HTTP) for ARTICBASTION to orchestrate scans:

- `CreateScan(scan_request)`  
  - Targets (CIDR ranges, host lists, asset groups).  
  - `scan_profile`.  
  - ROE and classification labels.  

- `GetScanStatus(scan_id)`  
  - Progress, error rates, packets/sec, etc.  

- `CancelScan(scan_id)`  

- `ListScans(filters)`  

### 5.2 Campaign & Case Binding

- All scans carry a `campaign_id` (or `case_id`) from ARTICBASTION.  
- FASTPORT attaches this ID to all emitted events so HDAIS and MEMSHADOW can correlate.

---

## 6. DSLLVM & Performance Strategy

### 6.1 Goals

- Keep FASTPORT as a **flagship DSLLVM-optimised engine**, demonstrating:  
  - Top-tier throughput (AVX-512, large batches).  
  - Strong hardening and safety (control-flow integrity, bounds checks).  
  - Fine-grained telemetry from DSLLVM passes.

### 6.2 Target Components

- **Rust core (fastport-core)**:
  - Core packet generation and parsing.  
  - SIMD-accelerated scan loops.  
  - Flood control and batching.

- **C / C++ helpers (optional)**:
  - Kernel-bypass or raw-socket shims where present.  
  - DPDK or AF\_XDP modules if added in the future.

### 6.3 Build Profiles

Define DSLLVM build modes aligned with the rest of the stack:

- `fastport-fast`
  - Optimised for central scanning nodes.  
  - AVX-512 first, AVX2 fallback.  
  - `-march=meteorlake -mtune=meteorlake` for your current hardware.  
  - Link-time optimisation (LTO) and profile-guided optimisation (PGO) using representative scans.

- `fastport-secure`
  - For edge / contested areas.  
  - Adds DSLLVM security passes: CFI, hardened allocation paths, stack protections.  
  - Slightly lower peak throughput but higher robustness.  

- `fastport-debug`
  - Built with sanitizers (ASan/UBSan) and high verbosity.  
  - For local QA and fuzz integration only.

### 6.4 Rust + DSLLVM Notes

Depending on how DSLLVM is integrated with Rust in your environment:

- If you have a Rust toolchain built against DSLLVM:
  - Export `RUSTC=/opt/dsllvm/bin/rustc` (path as appropriate).  
  - Add DSLLVM flags to `RUSTFLAGS` via environment or `.cargo/config.toml`.  

- If DSLLVM is available as a C/C++ compiler toolchain:
  - Build hot-path C/C++ fragments (packet loops, NIC shims) with DSLLVM.  
  - Expose them to Rust via FFI, keeping Rust as the orchestration + logic layer.  

Document the exact DSLLVM usage for this repo in `BUILD_DSLLVM.md` (see companion file).

---

## 7. Telemetry, Observability & QA

- Emit Prometheus metrics from the scanner core:
  - `fastport_packets_per_sec`  
  - `fastport_scan_duration_seconds`  
  - `fastport_events_emitted_total`  
  - `fastport_cves_found_total{severity=...}`  
- Write structured logs (JSON) for scan lifecycle events.  
- Provide synthetic QA scenarios:
  - Known-good lab ranges with seeded services and CVEs.  
  - Regression tests that validate throughput, correctness, and CVE tagging.

---

## 8. Milestones

1. **M1 – threat_event_v1 Output**
   - Implement canonical output mapping and streaming/batch writers.

2. **M2 – Scan Profiles & ICS-Safe Mode**
   - Implement named profiles and ICS/SCADA-safe scanning with guardrails.

3. **M3 – Enhanced Fingerprinting & CVE Matching**
   - TLS/HTTP/SSH/DB fingerprint upgrades and better CVE matching heuristics.

4. **M4 – ARTICBASTION Tasking API**
   - Implement CreateScan/GetScanStatus/CancelScan etc. wired into orchestration.

5. **M5 – DSLLVM Integration**
   - Introduce `fastport-fast` and `fastport-secure` build modes and measure gains.

6. **M6 – Telemetry + QA Lab**
   - Metrics, structured logs, and synthetic lab ranges for regression and benchmarking.
