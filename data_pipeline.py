from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits



def _make_native_endian(array: np.ndarray) -> np.ndarray:
    """Convert big-endian FITS arrays to native endianness for pandas."""
    if array.dtype.isnative:
        return array
    return array.astype(array.dtype.newbyteorder("=")).copy()


def _fits_hdu_to_dataframe(hdu) -> pd.DataFrame:
    """Convert a FITS table HDU to a pandas DataFrame.

    This handles structured arrays with nested vector columns like the SoLEXS
    `.pi` channel and counts arrays.
    """
    data = hdu.data
    if data is None or len(data) == 0:
        return pd.DataFrame()

    array = np.asarray(data)
    array = _make_native_endian(array)

    if array.dtype.names is None:
        if np.issubdtype(array.dtype, np.floating) and array.dtype == np.float64:
            array = array.astype(np.float32)
        return pd.DataFrame(array)

    columns = {}
    for name in array.dtype.names:
        col = array[name]
        if isinstance(col, np.ndarray) and col.ndim > 1:
            if np.issubdtype(col.dtype, np.floating) and col.dtype == np.float64:
                col = col.astype(np.float32)
            columns[name] = [row.tolist() for row in col]
        elif isinstance(col, np.ndarray) and col.dtype == object:
            processed_rows = []
            for row in col:
                if isinstance(row, np.ndarray):
                    if np.issubdtype(row.dtype, np.floating) and row.dtype == np.float64:
                        row = row.astype(np.float32)
                    processed_rows.append(row.tolist())
                else:
                    processed_rows.append(row)
            columns[name] = processed_rows
        else:
            if isinstance(col, np.ndarray) and np.issubdtype(col.dtype, np.floating) and col.dtype == np.float64:
                col = col.astype(np.float32)
            columns[name] = col

    df = pd.DataFrame(columns)
    for col_name in df.columns:
        if df[col_name].dtype == np.float64:
            df[col_name] = df[col_name].astype(np.float32)

    return df


def load_solexs_fits_table(file_path: str | Path, extension: int = 1, memmap: bool = True) -> pd.DataFrame:
    """Load the table from a SoLEXS FITS-like file (.lc, .gti, .pi)."""
    path = Path(file_path)
    with fits.open(path, memmap=memmap) as hdul:
        if extension >= len(hdul):
            raise IndexError(f"Extension {extension} not found in {path}")
        hdu = hdul[extension]
        return _fits_hdu_to_dataframe(hdu)


def extract_solexs_data(root_dir: str | Path, suffixes: tuple[str, ...] = (".lc", ".gti", ".pi"), memmap: bool = True) -> pd.DataFrame:
    """Extract all SoLEXS `.lc`, `.gti`, and `.pi` data under a root directory.

    Returns a single DataFrame containing rows from every table, with extra
    columns to identify the source file and HDU.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"SoLEXS root directory not found: {root}")

    rows = []
    records = []
    for file_path in sorted(root.rglob("*")):
        if file_path.suffix.lower() not in {s.lower() for s in suffixes}:
            continue

        data_type = file_path.suffix.lower().lstrip('.')
        with fits.open(file_path, memmap=memmap) as hdul:
            for extension, hdu in enumerate(hdul):
                if hdu.data is None:
                    continue
                df = _fits_hdu_to_dataframe(hdu)
                if df.empty:
                    continue
                df.insert(0, "source_file", str(file_path))
                df.insert(1, "data_type", data_type)
                df.insert(2, "hdu_name", hdu.name)
                df.insert(3, "hdu_index", extension)
                records.append(df)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def extract_solexs_file_data(file_path: str | Path, memmap: bool = True) -> pd.DataFrame:
    """Load one SoLEXS file into a DataFrame and add metadata columns."""
    file_path = Path(file_path)
    df = load_solexs_fits_table(file_path, memmap=memmap)
    if df.empty:
        return df
    df.insert(0, "source_file", str(file_path))
    df.insert(1, "data_type", file_path.suffix.lower().lstrip('.'))
    return df



# For HEL1OS data

def load_hel1os_fits_table(file_path: str | Path, extension: int = 1, memmap: bool = True) -> pd.DataFrame:
    """Load a table from a HEL1OS FITS file (.fits)."""
    path = Path(file_path)
    with fits.open(path, memmap=memmap) as hdul:
        if extension >= len(hdul):
            raise IndexError(f"Extension {extension} not found in {path}")
        hdu = hdul[extension]
        return _fits_hdu_to_dataframe(hdu)


def extract_hel1os_data(root_dir: str | Path, suffixes: tuple[str, ...] = (".fits",), memmap: bool = True) -> pd.DataFrame:
    """Extract all HEL1OS FITS table data under a root directory.

    Returns a concatenated DataFrame with metadata columns for the source file,
    HDU, and the HEL1OS sub-directory category (aux, cdte, czt, events).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"HEL1OS root directory not found: {root}")

    allowed_suffixes = {s.lower() for s in suffixes}
    records = []
    for file_path in sorted(root.rglob("*")):
        if file_path.suffix.lower() not in allowed_suffixes:
            continue

        rel_parts = file_path.relative_to(root).parts
        category = rel_parts[0] if len(rel_parts) >= 1 else "unknown"
        if len(rel_parts) > 1 and rel_parts[-2] in {"aux", "cdte", "czt", "events"}:
            category = rel_parts[-2]

        with fits.open(file_path, memmap=memmap) as hdul:
            for extension, hdu in enumerate(hdul):
                if hdu.data is None or not hasattr(hdu, "columns") or hdu.columns is None:
                    continue
                df = _fits_hdu_to_dataframe(hdu)
                if df.empty:
                    continue
                df.insert(0, "source_file", str(file_path))
                df.insert(1, "data_type", file_path.suffix.lower().lstrip('.'))
                df.insert(2, "hdu_name", hdu.name)
                df.insert(3, "hdu_index", extension)
                df.insert(4, "category", category)
                records.append(df)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def extract_hel1os_file_data(file_path: str | Path, memmap: bool = True) -> pd.DataFrame:
    """Load one HEL1OS FITS file into a DataFrame and add metadata columns."""
    file_path = Path(file_path)
    df = load_hel1os_fits_table(file_path, memmap=memmap)
    if df.empty:
        return df
    df.insert(0, "source_file", str(file_path))
    df.insert(1, "data_type", file_path.suffix.lower().lstrip('.'))
    return df
