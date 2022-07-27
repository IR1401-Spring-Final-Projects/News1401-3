import zipfile

with zipfile.ZipFile('models.zip', 'r') as file:
    file.extractall(path=".\\UI\\webui\\mir\\models")
