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
        #images_list.append(np.resize(cv2.imread(path + files[iter_image]), (50, 50, 3))[:, :, 0])
        img = image.resize(Size, Image.ANTIALIAS).getdata()
        images_list.append(img)
        filename_split = image_name.split(".")
        label.append([filename_split[0].replace("subject", ""), filename_split[1]])
    
    images = np.array(images_list)
    return images, label

def normalization(arr):
    if np.min(arr) < 0:
        arr -= np.min(arr)
    arr = arr / np.max(arr) * 255
    return arr


def EigenFace(Eigenface, path, Size, number = 25):
    for iter_face in range(0, number):
        arr = normalization(Eigenface.T[iter_face])

        print(arr)
        new_image = Image.fromarray(arr.reshape(Size)).convert('L')
        new_image.save(path + str(iter_face + 1) + ".png")



def Reconstruct(EigenFace, sample_image, Size):
    for iter_image in range(len(sample_image)):
        OutputImage = np.zeros(Size)
        for iter_eigen in range(len(EigenFace)):
            OutputImage += EigenFace[iter_eigen] * sample_image[iter_image]
        
        sample_image_norm = normalization(sample_image)
        OutputImage_norm = normalization(OutputImage)
        Origin_Image = Image.fromarray(sample_image_norm[iter_image].reshape(Size)).convert('L')
        Reconstruct_Image = Image.fromarray(OutputImage_norm.reshape(Size)).convert('L')

        Origin_Image.save("Reconstruction/Original" + str(iter_image + 1) +".png")
        Reconstruct_Image.save("Reconstruction/Reconstruct" + str(iter_image + 1) +".png")







