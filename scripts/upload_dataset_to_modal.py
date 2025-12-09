import modal

app = modal.App("kaggle-downloader")
vol = modal.Volume.from_name("geoguessr-data", create_if_missing=True)
kaggle_secret = modal.Secret.from_name("kaggle-secret")

@app.function(
    volumes={"/mnt": vol},
    timeout=3600,
    image=modal.Image.debian_slim().pip_install("kaggle"),
    secrets=[kaggle_secret]
)
def download_and_unzip():
    import subprocess
    import pathlib    
    print("Downloading dataset from Kaggle...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "geo-guessr-street-view-cs-gy-6643",
        "-p", "/mnt/data"
    ], check=True)
    
    print("Unzipping dataset...")
    import zipfile
    zip_path = pathlib.Path("/mnt/data/geo-guessr-street-view-cs-gy-6643.zip")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/mnt/data")
    
    print("Removing zip file...")
    zip_path.unlink()
    
    print("Done! Dataset is in /mnt/data")

@app.function(volumes={"/mnt": vol})
def ls(path: str = "/mnt/data"):
    import subprocess
    subprocess.run(["ls", "-lah", path])

@app.local_entrypoint()
def main():
    download_and_unzip.remote()