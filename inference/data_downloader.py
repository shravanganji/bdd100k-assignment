import gdown

import os



url = 'https://drive.google.com/uc?id=1O4GYEXtIZrvfHkOmPKzIcmg6A2X2DmOf'
output = 'data/bdd100k.zip'
gdown.download(url, output, quiet=False)


