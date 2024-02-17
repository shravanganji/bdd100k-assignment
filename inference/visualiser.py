import os
import numpy as np
from PIL import Image
from scalabel.label.io import load
from scalabel.vis.label import LabelViewer

# load prediction frames
frames = load('det/testing/det.json').frames

viewer = LabelViewer()
for frame in frames:
    img = np.array(Image.open(os.path.join('data/bdd100k/images/100k/val', frame.name)))
    viewer.draw(img, frame)
    viewer.save(os.path.join('vis_dir', frame.name))
