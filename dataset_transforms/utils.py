from enum import StrEnum
from pathlib import Path
import stat

class Dataset(StrEnum):
    RAW = "raw"
    ENRICHED = "enriched"
    CROPPED = "cropped"

transform_dataset_path = Path("intermediate-datasets")
dataset_path = Path("dataset")


def make_directory_writable(path: Path):
    path.chmod(stat.S_IRWXU)
