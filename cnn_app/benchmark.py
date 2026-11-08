"""Inference latency, throughput, and performance benchmarking module."""

import time
from typing import Dict
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total trainable and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple = (1, 1, 28, 28),
    warmup_runs: int = 50,
    benchmark_runs: int = 500,
) -> Dict[str, float]:
    """Benchmark inference latency (P50, P95, P99) and FPS throughput."""
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    latencies_ms = []

    # Benchmark loop
    with torch.no_grad():
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    fps = 1000.0 / np.mean(latencies_ms)

    return {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_fps": float(fps),
    }


def print_benchmark_report(bench_results: Dict[str, float], param_info: Dict[str, int]) -> None:
    """Print formatted performance metrics to terminal."""
    print("\n" + "=" * 55)
    print("        INFERENCE LATENCY & BENCHMARK REPORT")
    print("=" * 55)
    print(f"  Trainable Parameters : {param_info['trainable']:,}")
    print(f"  Total Parameters     : {param_info['total']:,}")
    print("-" * 55)
    print(f"  Mean Latency         : {bench_results['mean_latency_ms']:.3f} ms")
    print(f"  P50 Latency (Median) : {bench_results['p50_latency_ms']:.3f} ms")
    print(f"  P95 Latency          : {bench_results['p95_latency_ms']:.3f} ms")
    print(f"  P99 Latency          : {bench_results['p99_latency_ms']:.3f} ms")
    print(f"  Throughput           : {bench_results['throughput_fps']:.1f} inferences/sec")
    print("=" * 55 + "\n")
