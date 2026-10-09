import time
import numpy as np
from data_generator import DataGenerator
from numba_ops import compute_numba_metrics
from parquet_io import write_to_parquet, read_from_parquet, cleanup_parquet
from spark_processing import get_spark_session, process_with_spark


def run_benchmark(num_samples: int = 1_000_000):
    print(f"=== Starting Benchmark ({num_samples:,} samples) ===")

    # 1. Generate Data
    t0 = time.perf_counter()
    generator = DataGenerator()
    data = generator.generate(num_samples)
    t_gen = time.perf_counter() - t0
    print(f"[1] Data Generation:        {t_gen:.4f}s")

    # 2. Parquet Write Performance
    parquet_filename = "benchmark_temp.parquet"
    t0 = time.perf_counter()
    write_to_parquet(data, parquet_filename)
    t_write = time.perf_counter() - t0
    print(f"[2] Parquet IO (Write):     {t_write:.4f}s")

    # 3. Parquet Read & Numba Processing
    t0 = time.perf_counter()
    table = read_from_parquet(parquet_filename)
    f1 = table["feature_one"].to_numpy()
    f2 = table["feature_two"].to_numpy()

    # Trigger Numba JIT (warmup) then measure computation time
    _ = compute_numba_metrics(f1[:10], f2[:10])
    
    t_numba_start = time.perf_counter()
    mean_dist, std_dist = compute_numba_metrics(f1, f2)
    t_numba = time.perf_counter() - t_numba_start
    t_read_numba = time.perf_counter() - t0
    
    print(f"[3] Parquet Read + Numba:   {t_read_numba:.4f}s (Numba compute time: {t_numba:.4f}s)")
    print(f"    └─ Numba Output: Mean={mean_dist:.4f}, Std={std_dist:.4f}")

    # 4. PySpark Distributed Processing
    t0 = time.perf_counter()
    spark = get_spark_session()
    spark_metrics = process_with_spark(spark, parquet_filename)
    t_spark = time.perf_counter() - t0
    
    print(f"[4] PySpark Pipeline:       {t_spark:.4f}s")
    print(f"    └─ Spark Output: Mean={spark_metrics['avg_distance']:.4f}, Count={spark_metrics['total_records']}")

    # Clean up
    spark.stop()
    cleanup_parquet(parquet_filename)
    print("=== Benchmark Finished Successfully ===")


if __name__ == "__main__":
    run_benchmark(num_samples=500_000)