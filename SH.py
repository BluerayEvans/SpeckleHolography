import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir


def halfwidth(fouriertransform):
    """returns the width of the peak"""
    halfmax = max(fouriertransform) / 2
    for x in range(0, len(fouriertransform) - 1):
        if fouriertransform[x] >= halfmax:
            left = xvalues[x]
            break
    for x in range(len(fouriertransform) - 1, 0, -1):
        if fouriertransform[x] >= halfmax:
            right = xvalues[x]
            break
    return right - left


def widthlist():
    """generates list of widths of fourier transform"""
    widths = []
    for x in range(1, 5):
        imagearray = plt.imread("Images/NaughtyPics/" + str(x) + "mm_aperture.bmp")  # imports the image
        column = np.sum((imagearray[:, :]), 0)
        transformed_column = np.fft.fft(column)
        transformed_column = np.concatenate((transformed_column[372:-8], transformed_column[8:372]))
        abs_transformed_column = np.sqrt(transformed_column * np.conjugate(transformed_column))
        width = halfwidth(abs_transformed_column)
        widths.append(width)
    x = np.array([1, 2, 3, 4, 5])
    a, b = np.polyfit(x, widths, 1)
    plt.scatter(x, widths)
    plt.plot(x, a*x+b)

    plt.title('Aperture circumference vs half width max')
    plt.xlabel('Aperture circumference (mm)')
    plt.ylabel('Half width max')
    plt.show()
    return widths


def main():
    imagearray = plt.imread("Images/NaughtyPics/0" + circumference + "mm_aperture.bmp")  # imports the image
    #print(imagearray.shape)
    #imagearray = np.array(pd.read_csv('Images/data_result.csv'))
    column = np.sum((imagearray[:, :]), 0)

    print(column)

    transformed_column = np.fft.fft(column)
    print(transformed_column)
    abs_transformed_column2 = (transformed_column * np.conjugate(transformed_column)) ** 0.5

    transformed_column = np.concatenate((transformed_column[640:], transformed_column[1:640]))
    abs_transformed_column = abs(np.sqrt(transformed_column * np.conjugate(transformed_column)))
    #print(halfwidth(abs_transformed_column))
    plt.plot(xvalues[1:], abs_transformed_column)
    plt.title('Fourier transform of the image data for circumference: ' + circumference)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('Plots/absolute.png', bbox_inches='tight')
    plt.show()


#path = os.listdir("Images\Images3")
circumference = '1'
total_values = 1264
xvalues = np.linspace(-0.5, 0.5, total_values)

main()
print(widthlist())
