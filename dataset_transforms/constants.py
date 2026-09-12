from enum import StrEnum
from pathlib import Path

class Dataset(StrEnum):
    RAW = "raw"
    ENRICHED = "enriched"
    CROPPED = "cropped"

transform_dataset_path = Path("intermediate-datasets")
dataset_path = Path("dataset")
