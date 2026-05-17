from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".json", ".txt")
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "cp1251", "latin1")


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
            allowed = ", ".join(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
            raise ValueError(f"Unsupported file type. Allowed: {allowed}")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file_name).name.replace(" ", "_")
        target_path = self.upload_dir / safe_name
        target_path.write_bytes(content)
        return target_path


class DatasetProfiler:

    def profile(self, dataset_path: Path) -> DatasetProfile:
        dataframe = read_dataframe(dataset_path)
        sample = (
            dataframe.head(5)
            .where(pd.notnull(dataframe.head(5)), None)
            .to_dict(orient="records")
        )

        return DatasetProfile(
            file_name=dataset_path.name,
            extension=dataset_path.suffix.lower().lstrip("."),
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            column_names=[str(column) for column in dataframe.columns],
            dtypes={
                str(column): str(dtype)
                for column, dtype in dataframe.dtypes.items()
            },
            missing_values={
                str(column): int(value)
                for column, value in dataframe.isna().sum().items()
            },
            sample_records=sample,
        )


def read_dataframe(dataset_path: Path) -> pd.DataFrame:
    extension = dataset_path.suffix.lower()
    if extension == ".csv":
        return _read_csv_with_fallback(dataset_path)
    if extension == ".txt":
        return _read_csv_with_fallback(dataset_path, sep=None, engine="python")
    if extension == ".json":
        return pd.read_json(dataset_path)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)

    raise ValueError(f"Unsupported file extension: {extension}")


def _read_csv_with_fallback(dataset_path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(dataset_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as error:
            last_error = error

    message = (
        f"Unable to decode CSV file {dataset_path.name} with supported encodings: "
        f"{', '.join(CSV_ENCODINGS)}"
    )
    raise ValueError(message) from last_error
