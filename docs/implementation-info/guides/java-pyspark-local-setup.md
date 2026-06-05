# Local Java + PySpark Setup Guide (Mac & Windows)

| Field | Value |
|-------|-------|
| **Audience** | Developers running `notebooks/` locally before AWS Glue |
| **Repo env** | `.venv-notebooks` + kernel `Fashion Reco (notebooks)` |
| **Related** | [`README.md`](../../../README.md) · [`requirements-notebooks.txt`](../../../requirements-notebooks.txt) |

---

## 1. Why Java is required

PySpark is a Python API over **Apache Spark**, which runs on the **JVM**. When you create a `SparkSession`, PySpark starts a Java gateway process. No Java → no Spark.

This repo uses **local mode** only (`local[*]`): a single machine, all cores. You are **not** running a distributed cluster locally. AWS Glue provides Java and Spark at runtime in production.

---

## 2. Version compatibility (read this first)

Mismatch between Java and PySpark is the most common setup failure.

| PySpark | Java required | Notes |
|---------|---------------|-------|
| **3.4.x** | Java **8, 11, or 17** | **Pinned in this repo** — works on older Mac/Windows installs |
| **3.5+** | Java **17+** only | Default if you `pip install pyspark` without a pin |

This repo pins PySpark in `requirements-notebooks.txt`:

```txt
pyspark>=3.4.0,<3.5
```

**Recommendation:** Use **Java 17 (LTS)** on a fresh machine — it satisfies PySpark 3.4 and future 3.5+. If you already have Java 8 and want zero Java changes, keep the repo pin on 3.4.x.

---

## 3. Quick verification

After setup, all of these should succeed:

```bash
java -version
# openjdk version "17.x" or "1.8.x" ...

source .venv-notebooks/bin/activate   # Mac / Linux
# .venv-notebooks\Scripts\activate    # Windows

python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[*]').getOrCreate(); print('Spark', spark.version); spark.stop()"
```

Expected output includes a Spark version (e.g. `3.4.4`) with no `JAVA_GATEWAY_EXITED` or `UnsupportedClassVersionError`.

---

## 4. macOS setup

### 4.1 Install Java

**Option A — Homebrew + OpenJDK 17 (recommended for new setups)**

```bash
brew install openjdk@17
```

Add to your shell profile (`~/.zshrc` or `~/.bash_profile`):

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17 2>/dev/null || echo "/opt/homebrew/opt/openjdk@17")
export PATH="$JAVA_HOME/bin:$PATH"
```

Reload:

```bash
source ~/.zshrc
java -version
```

**Option B — Already have Java 8**

If `java -version` shows `1.8.x`, you can use the repo’s PySpark 3.4 pin without upgrading Java.

List installed JDKs:

```bash
/usr/libexec/java_home -V
```

Set a specific version:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

### 4.2 Python notebook environment

From the repo root (see [`README.md`](../../../README.md) for full steps):

```bash
python3 -m venv .venv-notebooks
source .venv-notebooks/bin/activate
pip install --upgrade pip
pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name fashion-reco-notebooks --display-name "Fashion Reco (notebooks)"
```

In VS Code / Cursor / Jupyter: select kernel **Fashion Reco (notebooks)**.

### 4.3 macOS-specific notes

- **`Unable to load native-hadoop library`** — harmless warning for local CSV work; ignore it.
- **Apple Silicon (M1/M2/M3)** — use Homebrew’s `openjdk@17` under `/opt/homebrew/opt/openjdk@17` on ARM.
- **Memory** — large H&M CSVs (~3 GB transactions) need driver memory; the sampling notebook sets `spark.driver.memory=4g`. Increase if you see OOM errors.

---

## 5. Windows setup

### 5.1 Install Java

**Recommended: Eclipse Temurin 17 (LTS)**

1. Download **JDK 17** from [Adoptium Temurin](https://adoptium.net/) (Windows x64 installer).
2. Run the installer; enable **“Set JAVA_HOME variable”** and **“Add to PATH”** if offered.
3. Open a **new** PowerShell or Command Prompt:

```powershell
java -version
echo $env:JAVA_HOME
```

**Manual JAVA_HOME** (if not set by installer):

1. Settings → System → About → Advanced system settings → Environment Variables.
2. New system variable:
   - Name: `JAVA_HOME`
   - Value: `C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot\` (your install path)
3. Add `%JAVA_HOME%\bin` to the system `Path`.
4. Restart the terminal and verify `java -version`.

### 5.2 Python notebook environment

From the repo root in PowerShell:

```powershell
python -m venv .venv-notebooks
.venv-notebooks\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name fashion-reco-notebooks --display-name "Fashion Reco (notebooks)"
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Select kernel **Fashion Reco (notebooks)** in your IDE.

### 5.3 Windows-specific notes

- Use **PowerShell** or **Windows Terminal**, not legacy `cmd`, for venv activation.
- Paths with spaces — keep the repo out of `Program Files`; a path like `C:\Users\you\Projects\fashion-recommendation-system` is fine.
- **Long paths** — if CSV writes fail, enable long path support in Windows or keep output under a short path.
- Same Hadoop native-library warning as Mac — safe to ignore locally.

---

## 6. Repo workflow after setup

| Step | Action |
|------|--------|
| 1 | Ensure `dataset/full/` contains H&M CSVs |
| 2 | Activate `.venv-notebooks` |
| 3 | Open `notebooks/stratified_user_sampling.ipynb` |
| 4 | Kernel: **Fashion Reco (notebooks)** |
| 5 | Run all cells → output in `dataset/sample/` |

Local Spark config in the notebook:

```python
SparkSession.builder.master("local[*]")  # single machine, all cores — not distributed
```

On AWS Glue later: same PySpark code, paths switch to S3 via env vars; Glue provides Java and Spark.

---

## 7. Troubleshooting

### `UnsupportedClassVersionError` / class file version 61.0 vs 52.0

**Cause:** PySpark 3.5+ installed but Java 8 on PATH.

**Fix (pick one):**

- Upgrade to Java 17 and set `JAVA_HOME`, **or**
- Stay on Java 8 and install the repo pin: `pip install "pyspark>=3.4.0,<3.5"`

### `JAVA_GATEWAY_EXITED` / Java gateway process exited before sending its port number

**Cause:** Java missing, wrong version, or `JAVA_HOME` not set.

**Fix:**

1. `java -version` must work in the **same terminal** you use for Jupyter.
2. Set `JAVA_HOME` to the JDK root (folder containing `bin/java`), not the `bin` folder itself.
3. Restart terminal / IDE after changing env vars.

### `command not found: java` (Mac)

Install JDK (§4.1) or fix PATH. IDE kernels **do not** always inherit shell profile — set `JAVA_HOME` in the IDE env or launch Jupyter from a terminal where `java -version` works.

### PySpark works in terminal but not in notebook kernel

The kernel may use a different Python or env.

1. Confirm kernel is **Fashion Reco (notebooks)** or **`.venv-notebooks`** (see below if missing in Cursor).
2. In a notebook cell:

```python
import sys, os
print(sys.executable)
print(os.environ.get("JAVA_HOME", "JAVA_HOME not set"))
```

`sys.executable` should point to `.venv-notebooks/.../python`.

### Kernel **Fashion Reco (notebooks)** not visible in Cursor / VS Code

The kernel is registered but Cursor often shows **Recommended** envs first (`mldl`, etc.) and hides Jupyter kernels.

**Fix (pick one):**

1. **Select Kernel** → **Jupyter Kernel…** (not Python Environments) → **Fashion Reco (notebooks)**
2. **Select Kernel** → **Python Environments** → **`.venv-notebooks`** (Python 3.11)
3. **Enter interpreter path…** → `<repo>/.venv-notebooks/bin/python`
4. **Cmd+Shift+P** → **Developer: Reload Window**

Re-register if needed (from repo root):

```bash
source .venv-notebooks/bin/activate
python -m ipykernel install --prefix="$(pwd)/.venv-notebooks" --name fashion-reco-notebooks --display-name "Fashion Reco (notebooks)"
python -m ipykernel install --user --name fashion-reco-notebooks --display-name "Fashion Reco (notebooks)"
```

Verify:

```bash
.venv-notebooks/bin/jupyter kernelspec list | grep fashion
```

### Out of memory on full H&M transactions

Increase driver memory when building the session:

```python
.config("spark.driver.memory", "8g")
```

Close other apps; the full transactions file is ~3.5 GB on disk.

### Slow first run

Spark JVM startup + CSV inference on 31M rows takes several minutes locally. This is expected; subsequent cells reuse the cached session.

---

## 8. Decision summary

| Goal | Java | PySpark |
|------|------|---------|
| Minimal change on existing Mac with Java 8 | Keep Java 8 | `3.4.x` (repo default) |
| New machine / long-term | Install Java **17** | `3.4.x` now; upgrade to 3.5+ later |
| Match future Glue 4.x / Spark 3.3+ in AWS | Java 17 locally | 3.4.x or 3.5+ after Java 17 |

---

## 9. References

- [Apache Spark Java compatibility](https://spark.apache.org/docs/latest/index.html)
- [PySpark installation](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
- Repo notebook: `notebooks/stratified_user_sampling.ipynb`
- Spec: `docs/superpowers/specs/2026-06-04-stratified-user-sampling-design.md`
