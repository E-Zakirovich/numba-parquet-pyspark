import os
import pyarrow as pa
import pyarrow.parquet as pq


def write_to_parquet(data: list[dict], filepath: str) -> str:
    """
    Converts raw dictionary records to a PyArrow Table and writes to disk in Parquet format.
    """
    table = pa.Table.from_pylist(data)
    pq.write_table(table, filepath, compression="snappy")
    return filepath


def read_from_parquet(filepath: str) -> pa.Table:
    """
    Reads a Parquet file back into a PyArrow columnar Table.
    """
    return pq.read_table(filepath)


def cleanup_parquet(filepath: str):
    """
    Removes temporary benchmark parquet files.
    """
    if os.path.exists(filepath):
        os.remove(filepath)