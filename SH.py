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
    for x in range(5, 13):
        imagearray = plt.imread("Images/Images3/" + str(x) + "mm circ test 3.bmp")  # imports the image
        column = imagearray[240, :]
        transformed_column = np.fft.fft(column)
        transformed_column = np.concatenate((transformed_column[372:], transformed_column[1:372]))
        abs_transformed_column = np.sqrt(transformed_column * np.conjugate(transformed_column))
        width = halfwidth(abs_transformed_column)
        widths.append(width)

    plt.scatter([5, 6, 7, 8, 9, 10, 11, 12], widths)

    plt.title('Aperture circumference vs half width max')
    plt.xlabel('Aperture circumference (mm)')
    plt.ylabel('Half width max')
    plt.show()
    return widths


def main():
    #imagearray = plt.imread("Images/Images3/" + circumference + "mm circ test 3.bmp")  # imports the image
    #print(imagearray.shape)
    imagearray = np.array(pd.read_csv('Images/data_result.csv'))
    column = imagearray[240, :]
    print(column)

    transformed_column = np.fft.fft(column)
    print(transformed_column)
    valover2 = (total_values/2)
    transformed_column = np.concatenate((transformed_column[256:], transformed_column[1:256]))
    abs_transformed_column = (transformed_column * np.conjugate(transformed_column)) ** 0.5
    print(halfwidth(abs_transformed_column))
    plt.plot(xvalues[1:], abs_transformed_column)
    plt.title('Fourier transform of the image data for circumference: ' + circumference)
    plt.xlabel('')
    plt.ylabel('')
    plt.xlim(0.5,1)
    plt.savefig('Plots/absolute.png', bbox_inches='tight')
    plt.show()


#path = os.listdir("Images\Images3")
circumference = '8'
total_values = 512
xvalues = np.linspace(0, 2.97, total_values+1)

main()
print(widthlist())
