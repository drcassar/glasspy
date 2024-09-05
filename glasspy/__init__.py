import io
import json
import shutil
import zipfile
from pathlib import Path

import requests
import sklearn
from platformdirs import user_data_dir

name = "glasspy"
version = "0.5.3"


def initial_config():
    current_versions = {
        "glasspy": version,
        "sklearn": sklearn.__version__,
    }

    config_dir = Path(user_data_dir("GlassPy"))
    config_file = config_dir / "versions.json"

    must_update = False

    if config_file.exists():
        with open(config_file, "r") as file:
            loaded_versions = json.load(file)

        for key in current_versions:
            if current_versions[key] != loaded_versions[key]:
                must_update = True

    else:
        must_update = True

    if must_update:

        print("[GlassPy] Initial installation or an update was detected.")
        print("[GlassPy] Downloading necessary files to your computer...")
        print(
            "[GlassPy] This is only required once and may take a few minutes."
        )

        try:
            shutil.rmtree(config_dir)
        except FileNotFoundError:
            pass

        config_dir.mkdir(parents=True, exist_ok=True)

        record_id = "13625981"
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url, timeout=60)
        record_data = response.json()
        url = record_data["files"][0]["links"]["self"]

        download = requests.get(url, timeout=3600)
        zip_file = io.BytesIO(download.content)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(config_dir)

        with open(config_file, "w") as file:
            json.dump(current_versions, file)

        print("[GlassPy] Download completed!")


initial_config()
