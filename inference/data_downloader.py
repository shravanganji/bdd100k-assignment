import gdown

import os



url = 'https://drive.google.com/uc?id=1kGQxFdTsXMTvx8lI6P4HGdYsKilZRRlh'
output = 'data/bdd100k.zip'
gdown.download(url, output, quiet=False)


