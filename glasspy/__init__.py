import io
import json
import os
import shutil
import tempfile
import zipfile
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import requests
import sklearn
from platformdirs import user_data_dir

name = "glasspy"

try:
    version_ = version("glasspy")

except PackageNotFoundError:
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
            version_ = pyproject_data["project"]["version"]
    else:
        version_ = "unknown"


def _read_versions(config_file):
    """Return the stored versions, or None if missing, corrupt or invalid."""
    try:
        with open(config_file, "r") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def initial_config():
    current_versions = {
        "glasspy": str(version_),
        "sklearn": str(sklearn.__version__),
    }

    config_dir = Path(user_data_dir("GlassPy"))
    config_file = config_dir / "versions.json"

    loaded_versions = _read_versions(config_file)

    must_update = loaded_versions is None or any(
        loaded_versions.get(key) != value for key, value in current_versions.items()
    )

    if not must_update:
        return

    print("[GlassPy] Initial installation or an update was detected.")
    print("[GlassPy] Downloading necessary files to your computer...")
    print("[GlassPy] This is only required once and may take a few minutes.")

    # Everything is prepared in a staging folder and only swapped in once the
    # download, the extraction, and the versions file are all complete. An
    # interrupted run therefore never leaves a half-written installation behind.
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="GlassPy_staging_",
            dir=config_dir.parent,
        )
    )

    try:
        record_id = "13625981"
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        url = response.json()["files"][0]["links"]["self"]

        download = requests.get(url, timeout=3600)
        download.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(download.content), "r") as zip_ref:
            zip_ref.extractall(staging)

        with open(staging / "versions.json", "w") as file:
            json.dump(current_versions, file)

        shutil.rmtree(config_dir, ignore_errors=True)
        os.replace(staging, config_dir)

    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print("[GlassPy] Download completed!")


# Set GLASSPY_SKIP_INIT=1 to skip the download step on import (for example,
# when building the documentation or running CI jobs that do not need the data).
if not os.environ.get("GLASSPY_SKIP_INIT"):
    initial_config()
