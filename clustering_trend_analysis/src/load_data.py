import json
import pandas as pd
from pathlib import Path
from typing import Union

def load_selected_fields(file_path: Union[str, Path], fields: list[str]) -> pd.DataFrame:
    """
    Load only the given fields from a JSON or CSV file.

    The CSV file is assumed to have its first row as column headers.

    Parameters
    ----------
    file_path : str | Path
        Path to the JSON or CSV file.
    fields : list of str
        The 4 field names to keep (must match the headers in the file).

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the specified fields.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load based on file type
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    elif file_path.suffix == ".csv":
        # ✅ pandas automatically uses the first row as headers
        df = pd.read_csv(file_path)

    else:
        raise ValueError("Unsupported file type. Please use JSON or CSV.")

    # Check that all requested fields exist
    missing_fields = [f for f in fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"These fields are missing in the file: {missing_fields}")

    # Keep only those 4 fields
    df = df[fields]

    return df
