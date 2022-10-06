import os
import numpy as np
import matplotlib.pyplot as plt

def transform_column(imagearray):
    column = np.sum((imagearray[:, :]), 0)
    transformed_column = np.fft.fft(column)
    half_values = total_values/2
    transformed_column = np.concatenate((transformed_column[int(half_values):total_values-(deletion-1)], transformed_column[int(deletion):int(half_values)]))
    abs_transformed_column = abs(np.sqrt(transformed_column * np.conjugate(transformed_column)))
    return abs_transformed_column


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
    for x in images:
        imagearray = plt.imread(filepath + x)  # imports the image
        abs_transformed_column = transform_column(imagearray)
        width = halfwidth(abs_transformed_column)
        widths.append(width)
    x = np.arange(0, len(images))
    a, b = np.polyfit(x, widths, 1)
    plt.scatter(x, widths)
    plt.plot(x, a*x+b)
    plt.title('Aperture circumference vs half width max')
    plt.xlabel('Aperture circumference (mm)')
    plt.ylabel('Half width max')
    plt.show()
    return widths


def main():
    imagearray = plt.imread(filepath + images[-1])  # imports the image
    abs_transformed_column = transform_column(imagearray)
    plt.plot(xvalues[int(deletion*2-1):], abs_transformed_column)
    plt.title('Fourier transform of the image data for circumference: ' + circumference)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('Plots/absolute.png', bbox_inches='tight')
    plt.show()


circumference = '6'
total_values = 739
xvalues = np.linspace(-0.5, 0.5, total_values)
deletion = 13
filepath = "Images/Images4/"
images = os.listdir(filepath)

main()
widthlist()
