import requests

HOST = '127.0.0.1'
PORT = '6969'

image_path = r"C:\Users\ASUS\Downloads\ANH QUET BAI THIv65\ANH_QUET_BAI_THI\ps3150u49.jpg"

def htmlTableAPI(image_path):
    # Open the image file in binary mode

    files = {
        "image": open(image_path, 'rb'),
    }

    resp = requests.post(f'http://{HOST}:{PORT}/predict', files=files)

    return resp.json()

htmlTableAPI(image_path)
