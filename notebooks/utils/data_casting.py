"""Pandas dtype casting utilities driven by configs/*/ml_types.yaml.

Usage
-----
Cast a raw DataFrame and save to Parquet::

    from utils.data_casting import cast_table, save_parquet

    articles = cast_table(articles_df, "articles", config_path="configs/data/ml_types.yaml")
    save_parquet(articles, "dataset/full_casted/articles.parquet")

The casting config uses real Pandas dtype strings so there is no translation layer:
  object         -- IDs, codes, free text (no-op if already object)
  category       -- low-cardinality nominal labels
  Int32          -- nullable integer (pandas capital-I, supports pd.NA)
  float32        -- continuous numeric values
  datetime64[ns] -- timestamp columns, handled via pd.to_datetime (writes as TIMESTAMP_NTZ)
  date           -- date-only columns (writes as Parquet date32 / INT32, like PySpark DateType)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import yaml


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a directory containing ``configs/`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir():
            return candidate
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_ml_types(config_path: str | Path | None = None, section: str | None = None) -> dict[str, str]:
    """Load column→dtype mapping from a ml_types YAML file.

    Parameters
    ----------
    config_path:
        Absolute or repo-relative path to the YAML file.
        Defaults to ``configs/data/ml_types.yaml`` relative to repo root.
    section:
        Top-level key inside the YAML (e.g. ``"articles"``, ``"features"``).
        When *None*, the entire file is returned as a flat dict (assumes
        the file has no sub-sections, or the caller wants the raw structure).

    Returns
    -------
    dict[str, str]
        ``{column_name: pandas_dtype_string}`` mapping.
    """
    repo_root = _find_repo_root()
    if config_path is None:
        config_path = repo_root / "configs" / "data" / "ml_types.yaml"
    else:
        p = Path(config_path)
        config_path = p if p.is_absolute() else repo_root / p

    raw = _load_yaml(Path(config_path))

    if section is not None:
        mapping = raw.get(section, {})
    else:
        mapping = raw

    if not isinstance(mapping, dict):
        raise ValueError(
            f"Expected a dict under section '{section}' in {config_path}, got {type(mapping)}"
        )
    return {col: str(dtype) for col, dtype in mapping.items()}


def cast_table(
    df: pd.DataFrame,
    section: str,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Cast a DataFrame's columns according to the ml_types YAML config.

    Only columns present in *both* the DataFrame and the config are cast;
    extra columns in either are silently ignored.

    Special handling:
    - ``datetime64[ns]`` uses ``pd.to_datetime()`` instead of ``astype``.
    - ``category`` on float columns (e.g. ``FN``, ``Active``) fills NaN with
      the string ``"unknown"`` before casting so NaN does not become a category
      level.
    - ``object`` columns are no-ops (already string-compatible).
    - ``Int32`` uses pandas nullable integer which supports ``pd.NA``.

    Parameters
    ----------
    df:
        Input DataFrame to cast.
    section:
        Top-level key in the YAML file (e.g. ``"articles"``, ``"features"``).
    config_path:
        Path to the YAML file. Defaults to ``configs/data/ml_types.yaml``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with columns cast to target dtypes. Original is unchanged.
    """
    dtype_map = load_ml_types(config_path=config_path, section=section)
    df = df.copy()

    for col, target_dtype in dtype_map.items():
        if col not in df.columns:
            continue

        current_dtype = str(df[col].dtype)

        if target_dtype == "object":
            if current_dtype != "object":
                df[col] = df[col].astype(str)

        elif target_dtype == "datetime64[ns]":
            if current_dtype != "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")

        elif target_dtype == "date":
            # Write as Parquet date32 (INT32) — same physical type as PySpark DateType.
            # PyArrow maps Python datetime.date objects → pa.date32() on write.
            # On read-back, pd.read_parquet() converts date32 → datetime64[ns] automatically.
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        elif target_dtype == "category":
            if df[col].dtype.kind == "f":
                df[col] = df[col].fillna("unknown").astype(str)
            df[col] = df[col].astype("category")

        elif target_dtype == "Int32":
            df[col] = df[col].astype("Int32")

        elif target_dtype == "float32":
            df[col] = df[col].astype("float32")

        else:
            df[col] = df[col].astype(target_dtype)

    return df


def _normalize_arrow_for_spark(table: pa.Table) -> pa.Table:
    """Replace large_string/large_binary with string/binary in all columns.

    PyArrow >= 0.16 defaults to large_string (LARGE_BYTE_ARRAY) for pandas object
    columns and dictionary<values=large_string> for Categorical columns. Spark's
    Parquet reader only supports BYTE_ARRAY (string), not LARGE_BYTE_ARRAY, so
    reading these files in Spark raises ColumnReaderImpl conversion errors.

    This function downcasts:
      - large_string  → string
      - large_binary  → binary
      - dictionary<values=large_string, ...> → dictionary<values=string, ...>
    """
    new_arrays: list[pa.Array] = []
    new_fields: list[pa.Field] = []

    for i, field in enumerate(table.schema):
        col = table.column(i)
        t = field.type

        if pa.types.is_large_string(t):
            col = col.cast(pa.string())
            new_fields.append(pa.field(field.name, pa.string(), nullable=field.nullable))
        elif pa.types.is_large_binary(t):
            col = col.cast(pa.binary())
            new_fields.append(pa.field(field.name, pa.binary(), nullable=field.nullable))
        elif pa.types.is_dictionary(t) and pa.types.is_large_string(t.value_type):
            target = pa.dictionary(t.index_type, pa.string(), t.ordered)
            col = col.cast(target)
            new_fields.append(pa.field(field.name, target, nullable=field.nullable))
        else:
            new_fields.append(field)

        new_arrays.append(col)

    return pa.table(
        {f.name: arr for f, arr in zip(new_fields, new_arrays)},
        schema=pa.schema(new_fields),
    )


def save_parquet(df: pd.DataFrame, path: str | Path, overwrite: bool = True) -> Path:
    """Write a DataFrame to Parquet, optionally removing an existing file or directory.

    Parquet output from Spark is often a *directory* of part files.
    This function handles both cases: if the target path is a directory it is
    removed entirely before writing the single-file Parquet.

    Writes Spark-compatible Parquet: large_string columns (PyArrow default for
    pandas object dtype) are downcast to string (BYTE_ARRAY + UTF8) before writing,
    since Spark's Parquet reader does not support LARGE_BYTE_ARRAY.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination file path (e.g. ``dataset/full_casted/articles.parquet``).
        Resolved relative to the repo root when not absolute.
    overwrite:
        When ``True`` (default), remove any existing file or directory at
        *path* before writing.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    repo_root = _find_repo_root()
    dest = Path(path) if Path(path).is_absolute() else repo_root / path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()

    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    arrow_table = _normalize_arrow_for_spark(arrow_table)
    pq.write_table(arrow_table, dest)
    return dest


def _ensure_partition_columns(df: pd.DataFrame, partition_cols: list[str]) -> pd.DataFrame:
    """Derive Hive partition keys when missing (Spark-compatible string formats)."""
    df = df.copy()
    needs_t_dat = any(c in partition_cols for c in ("year", "month"))
    if needs_t_dat and "t_dat" not in df.columns:
        raise ValueError("t_dat column required to derive year/month partition columns")

    if "year" in partition_cols and "year" not in df.columns:
        t_dat = pd.to_datetime(df["t_dat"])
        df["year"] = t_dat.dt.year.astype(str)

    if "month" in partition_cols and "month" not in df.columns:
        t_dat = pd.to_datetime(df["t_dat"])
        df["month"] = t_dat.dt.month.astype(str).str.zfill(2)

    if "snap_date" in partition_cols and "snap_date" in df.columns:
        df["snap_date"] = pd.to_datetime(df["snap_date"]).dt.date

    return df


def save_parquet_hive(
    df: pd.DataFrame,
    path: str | Path,
    partition_cols: list[str] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a DataFrame as a Spark-readable Hive-style Parquet dataset directory.

  Layout matches ``stratified_user_sampling.ipynb`` / ``dataset/dummy``:
    - Unpartitioned: ``{table}/part-*.parquet``, ``_SUCCESS``
    - Partitioned: ``{table}/year=YYYY/month=MM/part-*.parquet`` or ``snap_date=YYYY-MM-DD/``

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Output directory (e.g. ``dataset/sample_2000_users/articles``).
        Resolved relative to the repo root when not absolute.
    partition_cols:
        Hive partition column names. ``year``/``month`` are derived from ``t_dat``
        when absent; ``snap_date`` is normalized to ``date`` for date32 partitions.
    overwrite:
        When ``True`` (default), remove any existing file or directory at *path*
        before writing.

    Returns
    -------
    Path
        Absolute path of the written dataset directory.
    """
    repo_root = _find_repo_root()
    dest = Path(path) if Path(path).is_absolute() else repo_root / path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()

    partition_cols = list(partition_cols or [])
    if partition_cols:
        df = _ensure_partition_columns(df, partition_cols)

    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    arrow_table = _normalize_arrow_for_spark(arrow_table)

    pq.write_to_dataset(
        arrow_table,
        root_path=str(dest),
        partition_cols=partition_cols if partition_cols else None,
    )
    (dest / "_SUCCESS").touch()
    return dest


def save_labeled_data(
    df: pd.DataFrame,
    out_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save a labeled ranking dataset as Hive-partitioned Parquet by ``snap_date``.

    Writes Spark-compatible Parquet: ``snap_date`` as date32 (INT32) and string
    columns as BYTE_ARRAY (not large_string). Output layout::

        transactions_with_label/snap_date=YYYY-MM-DD/part-*.parquet

    When *filename* is set (legacy), writes a single file at ``out_dir/filename``
    instead of the Hive layout.

    Parameters
    ----------
    df:
        Labeled dataset with at least ``customer_id``, ``article_id``, ``label``,
        and ``snap_date`` columns.
    out_dir:
        Output directory (created if missing).
    filename:
        If provided, write a single Parquet file (legacy). Otherwise use Hive layout.

    Returns
    -------
    Path
        Absolute path of the written dataset directory or file.
    """
    df_out = df.copy()
    if filename is not None:
        if "snap_date" in df_out.columns:
            df_out["snap_date"] = pd.to_datetime(df_out["snap_date"]).dt.date
        return save_parquet(df_out, Path(out_dir) / filename)

    return save_parquet_hive(df_out, out_dir, partition_cols=["snap_date"])


def cast_and_save(
    df: pd.DataFrame,
    section: str,
    output_path: str | Path,
    config_path: str | Path | None = None,
    overwrite: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Convenience wrapper: cast *df* then save to Parquet.

    Parameters
    ----------
    df:
        Input DataFrame.
    section:
        YAML section key for the column dtype mapping.
    output_path:
        Destination Parquet path (file, not directory).
    config_path:
        Path to the ml_types YAML. Defaults to ``configs/data/ml_types.yaml``.
    overwrite:
        Remove existing file/directory at *output_path* before writing.

    Returns
    -------
    tuple[pd.DataFrame, Path]
        The casted DataFrame and the absolute path written to.
    """
    casted = cast_table(df, section=section, config_path=config_path)
    dest = save_parquet(casted, output_path, overwrite=overwrite)
    return casted, dest
