from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


@dataclass(frozen=True)
class DatasetProfile:
    file_name: str
    extension: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    sample_records: list[dict[str, Any]]

    def to_prompt_text(self) -> str:
        return (
            f"File: {self.file_name}\n"
            f"Format: {self.extension}\n"
            f"Shape: {self.rows} rows x {self.columns} columns\n"
            f"Columns: {', '.join(self.column_names)}\n"
            f"Dtypes: {self.dtypes}\n"
            f"Missing values: {self.missing_values}\n"
            f"Sample records: {self.sample_records}"
        )


class DatasetStorage:
    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir

    def save(self, file_name: str, content: bytes) -> Path:
        extension = Path(file_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type. Allowed: {allowed}")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file_name).name.replace(" ", "_")
        target_path = self.upload_dir / safe_name
        target_path.write_bytes(content)
        return target_path


class DatasetProfiler:
    def profile(self, dataset_path: Path) -> DatasetProfile:
        dataframe = read_dataframe(dataset_path)
        sample = dataframe.head(5).where(pd.notnull(dataframe.head(5)), None).to_dict(orient="records")
        return DatasetProfile(
            file_name=dataset_path.name,
            extension=dataset_path.suffix.lower().lstrip("."),
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            column_names=[str(column) for column in dataframe.columns],
            dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
            missing_values={str(column): int(value) for column, value in dataframe.isna().sum().items()},
            sample_records=sample,
        )


def read_dataframe(dataset_path: Path) -> pd.DataFrame:
    extension = dataset_path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(dataset_path)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    if extension == ".json":
        return pd.read_json(dataset_path)
    if extension == ".parquet":
        return pd.read_parquet(dataset_path)
    raise ValueError(f"Unsupported file extension: {extension}")
