import gdown

import os



url = 'https://drive.google.com/uc?id=1ovE01IRNoVbEnqX7q7uW66Am9G5NiKYv'
output = 'data/bdd100k.zip'
gdown.download(url, output, quiet=False)


