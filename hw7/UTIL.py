import numpy as np
import os
from PIL import Image
import cv2

## norm the output images to 0~255
def normalization(arr):
    if np.min(arr) < 0:
        arr -= np.min(arr)
    arr = arr / np.max(arr) * 255
    return arr

## Using Image in PIL to plot the output imates
def EigenFace(Eigenface, path, Size, number = 25):
    for iter_face in range(0, number):
        arr = normalization(Eigenface[iter_face])

        new_image = Image.fromarray(arr.reshape(Size)).convert('L')
        new_image.save(path + str(iter_face + 1) + ".png")



def Reconstruct(EigenFace, sample_image, Size, Path):
    EigenFace = EigenFace.T

    ## Catch the important eigenface on each sample images
    for iter_image in range(len(sample_image)):
        OutputImage = np.zeros(Size[0] * Size[1])
        for iter_eigen in range(len(EigenFace)):
            OutputImage += EigenFace[iter_eigen] * sample_image[iter_image]
        
        ## Output the original and reconstruct images, 
        sample_image_norm = normalization(sample_image)
        OutputImage_norm = normalization(OutputImage)
        Origin_Image = Image.fromarray(sample_image_norm[iter_image].reshape(Size)).convert('L')
        Reconstruct_Image = Image.fromarray(OutputImage_norm.reshape(Size)).convert('L')

        Origin_Image.save(Path + "Origin/" + str(iter_image + 1) +".png")
        Reconstruct_Image.save(Path + "Reconstruction/" + str(iter_image + 1) +".png")


def KNN(type_A, k, images, EigenFace, proj_train_image, label, test_images, test_label):
    mean = np.mean(images, axis = 0)
    ## project images on EigenFace or fisherface
    if type_A == "PCA":
        ## PCA
        proj_test_image = ((test_images - mean) @ EigenFace.T) ## 30 * 25
    else:
        ## LDA
        proj_test_image = ((test_images) @ EigenFace.T) ## 30 * 25
    just_label = np.array(label)

    ## normalizate all of the date to 0~255
    proj_train_image_norm = np.zeros((proj_train_image.shape))
    for iter_norm in range(len(proj_train_image_norm)):
        proj_train_image_norm[iter_norm] = normalization(proj_train_image[iter_norm])


    ## choose the k nearest neighbors to decide which class it is.
    it = 0
    for iter_test_image in range(len(proj_test_image)):
        distance = np.zeros(len(label))
        for iter_dis in range(len(label)):
            distance[iter_dis] = np.linalg.norm(proj_train_image_norm[iter_dis] - proj_test_image[iter_test_image])
        
        outcome = list(just_label[np.argsort(distance)[:k]][:, 1])
        ans = outcome[np.argmax([outcome.count(outcome[i]) for i in range(len(outcome))])]
        
        ## calculate the precise
        if ans == test_label[iter_test_image][1]:
            it += 1
    print("outcome:", it)



