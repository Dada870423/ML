import numpy as np
import os
from PIL import Image
import cv2

def Readfile(path, Size):
    files = os.listdir(path)
    images_list = list()
    label = list()
    for image_name in files:
        image = Image.open(path + image_name)
        img = image.resize(Size, Image.ANTIALIAS).getdata()
        images_list.append(img)
        filename_split = image_name.split(".")
        label.append([filename_split[0].replace("subject", ""), filename_split[1]])
    
    images = np.array(images_list)
    return images, label