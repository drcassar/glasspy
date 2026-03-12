import requests
from pathlib import Path


def get_from_zenodo(ID, filename, savepath, verbose=True):
    """Download a single file from a Zenodo repository.

    Skips the download if the file already exists at the given path.
    Files are streamed in chunks to handle large downloads efficiently.

    Args:
      ID (str or int):
        Zenodo record ID. Found in the record's URL, e.g.,
        `zenodo.org/records/1234567` → ID is `1234567`.
      filename (str):
        Name of the file to download from the record.
      savepath (str or Path):
        Local path where the file will be saved.
      verbose (bool, optional):
        If `True`, prints progress messages during the download. Defaults to `True`.

    Returns:
      None: Returns immediately without downloading if the file already exists
        at `savepath`.

    Raises:
      requests.exceptions.HTTPError:
        If the HTTP request to Zenodo returns an error status code.
      StopIteration:
        If no file matching `filename` is found in the record.
    """

    if not isinstance(savepath, Path):
        savepath = Path(savepath)

    if savepath.is_file():
        return

    if verbose:
        print(f"[GlassPy] Downloading the file '{filename}'.")
        print("[GlassPy] This is only required once and may take a few minutes.")

    api_url = f"https://zenodo.org/api/records/{ID}"
    response = requests.get(api_url, timeout=60)
    files = response.json()["files"]
    file_info = next(f for f in files if f["key"] == filename)
    download_url = file_info["links"]["self"]

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(savepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if verbose:
        print("[GlassPy] Download completed!")
