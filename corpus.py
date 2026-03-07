import re
import urllib.request
import zipfile
import os


def tokenize(text: str) -> list[str]:

    text = text.lower()
    # remove anything that is not a lowercase letter or whitespace
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return tokens


def load_corpus(max_chars: int = 10_000_000) -> list[str]:

    text8_path = "text8"

    if not os.path.exists(text8_path):
        print("Downloading text8...")
        url = "http://mattmahoney.net/dc/text8.zip"
        urllib.request.urlretrieve(url, "text8.zip")
        with zipfile.ZipFile("text8.zip", "r") as zf:
            zf.extractall(".")
        os.remove("text8.zip")
        print("Download complete.")

    with open(text8_path, "r") as f:
        text = f.read(max_chars)

    tokens = tokenize(text)
    return tokens
