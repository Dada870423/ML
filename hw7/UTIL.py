import numpy as np
import os
from PIL import Image
import cv2


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



def Reconstruct(EigenFace, sample_image, Size, Path):
    EigenFace = EigenFace.T
    print("sample_image", sample_image.shape)
    print("EigenFace", EigenFace[:, 0].shape)

    for iter_image in range(len(sample_image)):
        OutputImage = np.zeros(Size[0] * Size[1])
        for iter_eigen in range(len(EigenFace)):
            OutputImage += EigenFace[iter_eigen] * sample_image[iter_image]
        
        sample_image_norm = normalization(sample_image)
        OutputImage_norm = normalization(OutputImage)
        Origin_Image = Image.fromarray(sample_image_norm[iter_image].reshape(Size)).convert('L')
        Reconstruct_Image = Image.fromarray(OutputImage_norm.reshape(Size)).convert('L')

        Origin_Image.save(Path + "Origin/" + str(iter_image + 1) +".png")
        Reconstruct_Image.save(Path + "Reconstruction/" + str(iter_image + 1) +".png")



def Sort_Norm_eigen(eigen_values_unsorted, eigen_vectors_unsorted, mean, images):
    Eigen_index = np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index]

    print("mean:", mean.shape)
    print("Eigen_Vector", Eigen_Vector.shape)

    print("(images - mean).T ", ((images - mean).T).shape)

    print("image", images.shape)

    eigen_vec = ((images - mean) @ Eigen_Vector)[:, :25]

    print(eigen_vec.shape, "eigen_vec")

    for iter_vector in range(25):
        eigen_vec[:, iter_vector] /= np.linalg.norm(eigen_vec[:, iter_vector])

    print("eigen_vec", eigen_vec.shape)
    return eigen_vec







