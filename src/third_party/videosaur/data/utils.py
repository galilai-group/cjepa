import math
import os
import pathlib
import re
import urllib.parse

import requests
import tqdm


def get_default_dir(subdir):
    try:
        import third_party.videosaur.videosaur.data

        default_dir = third_party.videosaur.videosaur.data.get_data_root_dir()
    except ImportError:
        default_dir = ".."

    return os.path.join(default_dir, subdir)


def download_file(url: str, dest_dir: str) -> pathlib.Path:
    """Download file to location and return path to location.

    Adapted from https://stackoverflow.com/a/10744565 and https://stackoverflow.com/a/53299682.
    """
    response = requests.get(url, stream=True)

    if "Content-Disposition" in response.headers:
        file_name = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
    else:
        url_path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        file_name = pathlib.Path(url_path).name

    if "Content-Length" in response.headers:
        file_size_kb = math.ceil(int(response.headers["Content-Length"]) / 1024)
    else:
        file_size_kb = None

    dest_file = pathlib.Path(dest_dir) / (file_name + ".tmp")
    with open(dest_file, "wb") as handle:
        for data in tqdm.tqdm(response.iter_content(chunk_size=1024), unit="kB", total=file_size_kb):
            handle.write(data)

    dest_file = dest_file.rename(dest_file.with_name(file_name))

    return dest_file
