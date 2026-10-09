from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_spark_session(app_name: str = "NumbaParquetSparkBench") -> SparkSession:
    """
    Initializes a local PySpark session.
    """
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.master", "local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def process_with_spark(spark: SparkSession, parquet_path: str):
    """
    Loads Parquet data into Spark DataFrame and performs aggregated statistics.
    """
    df = spark.read.parquet(parquet_path)

    # Distributed computation: distance calculation and aggregations
    processed_df = df.withColumn(
        "distance", F.sqrt(F.pow(F.col("feature_one"), 2) + F.pow(F.col("feature_two"), 2))
    )

    metrics = processed_df.select(
        F.avg("distance").alias("avg_distance"),
        F.stddev("distance").alias("std_distance"),
        F.count("id").alias("total_records"),
    ).collect()

    return metrics[0]