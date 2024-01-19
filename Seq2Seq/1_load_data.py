import os
import zipfile
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0'
}

def download_zip(url, zip_file):
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'ZIP file downloaded to {zip_file}')
    else:
        print(f'Failed to download. HTTP Response Code: {response.status_code}')

if __name__ == '__main__':

    # download url
    url = 'http://www.manythings.org/anki/'
    # select language zip 
    zip_file = 'kor-eng.zip'

    path = os.getcwd()
    zip_file_path = os.path.join(path, zip_file)

    if not os.path.isfile(zip_file_path):
        download_zip(url+zip_file, zip_file)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_f:
        zip_f.extractall(path)

