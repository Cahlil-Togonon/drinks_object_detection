import os
import gdown
import shutil

def download_model(model_path):
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=1hJCwL0hy-USoPuUd1JMv1IYpuQs1nifT'    # fasterrcnn_model_drinks_Epoch9.pt
        gdown.download(url, model_path, quiet=False)

def download_dataset(data_path):
    if not os.path.exists(data_path):
        url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'    # drinks.tar.gz
        gdown.download(url, quiet=False)
        print("Unpacking Drinks Dataset...")
        shutil.unpack_archive('drinks.tar.gz')
        print("Done!")
        