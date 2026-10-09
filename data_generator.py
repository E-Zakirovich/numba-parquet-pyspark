"""
data_generator.py
~~~~~~~~~~~~~~~~~

Generates a finite dataset of samples (for saving to Parquet).
"""

import random
from datetime import datetime


class DataGenerator:

    def __init__(self, low: float = 1, high: float = 20):
        self.low = low
        self.high = high

    def generate_sample(self, id: int) -> dict:
        return {
            "id": id,
            "time": datetime.now(),
            "feature_one": random.uniform(self.low, self.high),
            "feature_two": random.uniform(self.low, self.high),
        }

    def validate_sample(self, sample: dict) -> bool:
        return None not in sample.values()

    def generate(self, n: int) -> list[dict]:
        rows = []
        for i in range(1, n + 1):
            sample = self.generate_sample(i)
            if self.validate_sample(sample):
                rows.append(sample)
        return rows