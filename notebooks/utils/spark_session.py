"""Local / Glue SparkSession setup for notebooks — no src/ imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pyspark import SparkContext
from pyspark.sql import SparkSession

from utils.config_loader import find_repo_root, is_glue_env


def ensure_notebooks_path(start: Path | None = None) -> Path:
    """Add ``notebooks/`` to ``sys.path`` when the kernel cwd is repo root or notebooks/.

    Parameters
    ----------
    start : Path | None
        Directory to search from. Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        Resolved notebooks directory.
    """
    base = (start or Path.cwd()).resolve()
    candidates = (
        base,
        base / "notebooks",
        base.parent / "notebooks",
    )
    for candidate in candidates:
        if (candidate / "utils" / "spark_session.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    return base


def _detect_java_home() -> None:
    """Set JAVA_HOME on macOS when the IDE kernel did not inherit it."""
    if os.environ.get("JAVA_HOME"):
        return
    for java_home_cmd in (
        ["/usr/libexec/java_home", "-v", "17"],
        ["/usr/libexec/java_home", "-v", "1.8"],
        ["/usr/libexec/java_home"],
    ):
        try:
            os.environ["JAVA_HOME"] = subprocess.check_output(
                java_home_cmd, text=True, stderr=subprocess.DEVNULL
            ).strip()
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue


def _configure_windows_hadoop(repo_root: Path) -> str | None:
    """Configure HADOOP_HOME for local Parquet writes on Windows."""
    if sys.platform != "win32":
        return os.environ.get("HADOOP_HOME")

    def _apply_hadoop_home(hadoop_home: str) -> str:
        os.environ["HADOOP_HOME"] = hadoop_home
        bin_dir = str((Path(hadoop_home) / "bin").resolve())
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return hadoop_home

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and (Path(hadoop_home) / "bin" / "winutils.exe").exists():
        return _apply_hadoop_home(hadoop_home)

    bundled = repo_root / ".hadoop-win"
    if (bundled / "bin" / "winutils.exe").exists():
        return _apply_hadoop_home(str(bundled.resolve()))

    raise RuntimeError(
        "HADOOP_HOME is unset and .hadoop-win/bin/winutils.exe is missing. "
        "Parquet writes fail on Windows without winutils. "
        "See docs/implementation-info/guides/java-pyspark-local-setup.md §5.4 "
        "or set HADOOP_HOME to a folder containing bin/winutils.exe."
    )


def _reset_stale_spark() -> None:
    """Drop Python-side Spark singletons when the JVM gateway is dead."""
    SparkSession._instantiatedSession = None
    SparkContext._active_spark_context = None
    SparkContext._gateway = None
    SparkContext._jvm = None


def create_spark_session(
    app_name: str,
    *,
    is_glue: bool | None = None,
    driver_memory: str = "4g",
    shuffle_partitions: int = 8,
    master: str = "local[*]",
    log_level: str = "WARN",
    pin_python: bool = False,
) -> SparkSession:
    """Create or reuse a SparkSession for notebook workloads.

    On AWS Glue, returns the managed active session. Locally, configures Java,
    Windows Hadoop shims, and builds a ``local[*]`` driver.

    Parameters
    ----------
    app_name : str
        Spark application name.
    is_glue : bool | None
        When True, reuse Glue's session. When None, auto-detect via
        ``AWS_EXECUTION_ENV``.
    driver_memory : str
        ``spark.driver.memory`` for local runs.
    shuffle_partitions : int
        ``spark.sql.shuffle.partitions`` for local runs.
    master : str
        Spark master URL for local runs.
    log_level : str
        SparkContext log level (e.g. ``WARN``).
    pin_python : bool
        When True, pin ``PYSPARK_PYTHON`` / ``PYSPARK_DRIVER_PYTHON`` to
        ``sys.executable`` (helps on Windows kernels).

    Returns
    -------
    SparkSession
        Active Spark session.
    """
    glue = is_glue if is_glue is not None else is_glue_env()
    if glue:
        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("Glue runtime expected an active SparkSession")
        session.sparkContext.setLogLevel(log_level)
        return session

    if pin_python:
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    _detect_java_home()
    repo_root = find_repo_root()
    hadoop_home = _configure_windows_hadoop(repo_root)

    try:
        active = SparkSession.getActiveSession()
    except AssertionError:
        active = None
    if active is not None:
        try:
            active.sparkContext._jsc.sc().version()
        except Exception:
            _reset_stale_spark()
    elif SparkSession._instantiatedSession is not None or SparkContext._active_spark_context is not None:
        _reset_stale_spark()

    if sys.platform == "win32":
        active = SparkSession.getActiveSession()
        if active is not None:
            active.stop()
            _reset_stale_spark()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Allow reading Parquet files where date columns (INT32) sit alongside
        # timestamp_ntz columns (INT64) across partitions — happens when some
        # partitions were written by Pandas (datetime64→timestamp_ntz) and others
        # by PySpark DateType (INT32).  The non-vectorized reader coerces freely.
        .config("spark.sql.parquet.enableVectorizedReader", "false")
    )
    if hadoop_home:
        builder = (
            builder.config("spark.hadoop.hadoop.home.dir", hadoop_home)
            .config("spark.hadoop.io.native.lib.available", "false")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark
