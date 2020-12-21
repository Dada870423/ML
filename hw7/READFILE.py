import numpy as np
import os
import cv2

def Readfile(path):
    files = os.listdir(path)
    images_list = list()
    label = list()
    for iter_image in range(len(files)):
        images_list.append(np.resize(cv2.imread(path + files[iter_image]), (50, 50, 3))[:, :, 0])
        filename_split = files[iter_image].split(".")
        label.append([filename_split[0].replace("subject", ""), filename_split[1]])
    
    images = np.array(images_list)
    return images, label