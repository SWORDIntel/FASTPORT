//! FastPort Core - High-Performance Async Port Scanner
//!
//! Rust-based scanner core with AVX-512/AVX2 SIMD optimizations
//! and P-core thread pinning for maximum performance.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::runtime::Runtime;
use parking_lot::RwLock;

mod async_scanner;
mod packet_processor;
mod simd_scanner;
mod thread_pinning;

pub use async_scanner::*;
pub use packet_processor::*;
pub use simd_scanner::*;
pub use thread_pinning::*;

// Include compile-time CPU feature detection
include!(concat!(env!("OUT_DIR"), "/cpu_features.rs"));

/// Scanner statistics shared across threads
#[derive(Debug, Default, Clone)]
pub struct ScanStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub ports_open: u32,
    pub ports_closed: u32,
    pub ports_filtered: u32,
    pub scan_duration_ms: u64,
}

/// Global scanner instance with thread-safe statistics
pub struct FastPortScanner {
    runtime: Arc<Runtime>,
    stats: Arc<RwLock<ScanStats>>,
    worker_count: usize,
}

impl FastPortScanner {
    /// Create new scanner with P-core pinning
    pub fn new(worker_count: Option<usize>) -> anyhow::Result<Self> {
        let pcore_count = get_pcore_count();
        let workers = worker_count.unwrap_or(pcore_count);

        // Create multi-threaded runtime with P-core affinity
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .thread_name("fastport-worker")
            .enable_all()
            .on_thread_start(move || {
                // Pin to P-cores on startup
                if let Err(e) = pin_to_pcore() {
                    eprintln!("Warning: Failed to pin thread to P-core: {}", e);
                }
            })
            .build()?;

        Ok(Self {
            runtime: Arc::new(runtime),
            stats: Arc::new(RwLock::new(ScanStats::default())),
            worker_count: workers,
        })
    }

    /// Get SIMD variant being used
    pub fn simd_variant(&self) -> &'static str {
        get_simd_variant()
    }

    /// Get worker count
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Get current statistics
    pub fn get_stats(&self) -> ScanStats {
        self.stats.read().clone()
    }
}

/// Python module initialization
#[pymodule]
fn fastport_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastPortScanner>()?;
    m.add_function(wrap_pyfunction!(get_cpu_features, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_simd, m)?)?;
    Ok(())
}

/// Python wrapper for FastPortScanner
#[pyclass(name = "FastPortScanner")]
struct PyFastPortScanner {
    inner: Arc<FastPortScanner>,
}

#[pymethods]
impl PyFastPortScanner {
    #[new]
    fn new(worker_count: Option<usize>) -> PyResult<Self> {
        let scanner = FastPortScanner::new(worker_count)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(scanner),
        })
    }

    /// Async scan method callable from Python
    fn scan<'py>(
        &self,
        py: Python<'py>,
        target: String,
        ports: Vec<u16>,
        timeout_ms: u64,
    ) -> PyResult<&'py PyAny> {
        let scanner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let results = scanner
                .runtime
                .spawn(async move {
                    async_scan_ports(&target, &ports, timeout_ms).await
                })
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))??;

            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("target", target)?;
                dict.set_item("ports_open", results.len())?;
                dict.set_item("results", results)?;
                Ok(dict.into())
            })
        })
    }

    fn get_simd_variant(&self) -> &'static str {
        self.inner.simd_variant()
    }

    fn get_worker_count(&self) -> usize {
        self.inner.worker_count()
    }

    fn get_stats(&self) -> PyResult<String> {
        let stats = self.inner.get_stats();
        Ok(format!(
            "Packets: {}/{}, Ports: {} open, {} closed, Duration: {}ms",
            stats.packets_received,
            stats.packets_sent,
            stats.ports_open,
            stats.ports_closed,
            stats.scan_duration_ms
        ))
    }
}

/// Get CPU features detected at compile time
#[pyfunction]
fn get_cpu_features() -> PyResult<String> {
    let variant = get_simd_variant();
    let pcores = get_pcore_count();
    Ok(format!(
        "SIMD: {}, P-cores: {}, AVX-512: {}, AVX2: {}",
        variant, pcores, AVX512_ENABLED, AVX2_ENABLED
    ))
}

/// Benchmark SIMD performance
#[pyfunction]
fn benchmark_simd() -> PyResult<String> {
    let iterations = 1_000_000;
    let start = std::time::Instant::now();

    // Benchmark packet processing with SIMD
    for _ in 0..iterations {
        let _ = simd_process_packet(&[0u8; 64]);
    }

    let duration = start.elapsed();
    let ops_per_sec = (iterations as f64 / duration.as_secs_f64()) / 1_000_000.0;

    Ok(format!(
        "SIMD Variant: {}\nProcessed {} packets in {:?}\nThroughput: {:.2}M packets/sec",
        get_simd_variant(),
        iterations,
        duration,
        ops_per_sec
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scanner_creation() {
        let scanner = FastPortScanner::new(None).unwrap();
        assert!(scanner.worker_count() > 0);
        assert!(!scanner.simd_variant().is_empty());
    }

    #[test]
    fn test_cpu_features() {
        let features = get_cpu_features().unwrap();
        assert!(features.contains("SIMD"));
    }
}
