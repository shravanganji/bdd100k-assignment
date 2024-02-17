import gdown

url = 'https://drive.google.com/uc?id=1WS-Pqje2d42mjxj6WJuVrrUtsrj6R9qE'
output = 'csv_data.zip'
gdown.download(url, output, quiet=False)


