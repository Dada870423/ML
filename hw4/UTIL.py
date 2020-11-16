import sys
import math
import numpy as np

def init_data(image_file):
    Image_fptr = open(image_file, "rb")

    ## init image file
    Image_fptr.read(4) ## magic number
    Image_fptr.read(4) ## number of images
    Image_fptr.read(4) ## number of rows
    Image_fptr.read(4) ## number of columns

    return Image_fptr


def get_label(fptr):
    label = int.from_bytes(fptr.read(1), byteorder = 'big')
    return label


def get_pixel(fptr):
    pixel = int.from_bytes(fptr.read(1), byteorder = 'big')
    return pixel

def norm_probability(probability):
    total = 0.0
    for iter_i in range(len(probability)):
        total = total + probability[iter_i]
    for iter_i in range(len(probability)):
        probability[iter_i] = float(float(probability[iter_i]) / float(total))
    #print(total)
    return probability


def printProgress(iteration, total, prefix = "", suffix = "", decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()

def compare(predict_probability, Ans, fptr):
    prediction = np.argmin(predict_probability)
    fptr.write("\nPrediction: " + str(prediction) + ", Ans: " + str(Ans))
    fptr.write("\nPosterior (in log scale):")
    for digit in range(10):
        fptr.write(str(digit) + ": " + str(predict_probability[digit]) + "\n")
    return int(prediction != Ans)

