import os
import gdown
import zipfile

data_id = "1qGzgRRcm359d6GkbtW03Fi-nL-1w8Kku"
output = "data/data.zip"

gdown.download(id=data_id, output=output, quiet=False)
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("data/")

os.remove("data/data.zip")
