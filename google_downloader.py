import os
import gdown
import shutil

def download_model(model_path):
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/u/0/uc?id=1-JLadMe-gu89P808wb87w2dONlwmYV8g'
        gdown.download(url, model_path, quiet=False)

def download_dataset(data_path):
    if not os.path.exists(data_path):
        url = 'https://drive.google.com/u/0/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
        gdown.download(url, quiet=False)
        print("Unpacking Drinks Dataset...")
        shutil.unpack_archive('drinks.tar.gz')
        print("Done!")
        