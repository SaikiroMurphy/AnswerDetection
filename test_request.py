import os

from loguru import logger
import requests

HOST = '127.0.0.1'
PORT = '6969'

image_path = r"C:\Users\ASUS\Downloads\ANH QUET BAI THIv6\ANH QUET BAI THI"


def htmlTableAPI(image_path):
    # Open the image file in binary mode

    files = {
        "image": open(image_path, 'rb')
    }

    resp = requests.post(f'http://{HOST}:{PORT}/predict', files=files)

    return resp.json()


for file in os.listdir(image_path):
    if file.endswith(".jpg"):
        logger.info(htmlTableAPI(os.path.join(image_path,file)))
