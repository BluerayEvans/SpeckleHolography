import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals._pilutil import toimage
from scipy import optimize


def transform_column(imagearray):
    column = np.sum((imagearray[:, :]), 0)
    transformed_column = np.fft.fft(column)
    half_values = total_values / 2
    transformed_column = np.concatenate((transformed_column[int(half_values):total_values - (deletion - 1)],
                                         transformed_column[int(deletion):int(half_values)]))
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
    a, b = np.polyfit(np.delete(x, [1,6,7,8]), np.delete(widths, [1,6,7,8]), 1)
    plt.scatter(x, widths)
    plt.plot(x, a * x + b)
    plt.title('Aperture circumference vs half width max')
    plt.xlabel('Aperture circumference (mm)')
    plt.ylabel('Half width max (k value)')
    plt.ylim(0, 0.2)
    plt.show()
    return widths


def practicemain():
    imagearray = plt.imread(filepath + images[-2])  # imports the image
    abs_transformed_column = transform_column(imagearray)
    plt.plot(xvalues[int(deletion * 2 - 1):], abs_transformed_column)
    plt.title('F.T. of the image data for circumference: ' + images[-2])
    plt.xlabel('k value')
    plt.ylabel('Summed intensity along column')
    plt.savefig('Plots/absolute.png', bbox_inches='tight')
    plt.show()

def fit_function(x, a, b, c):
    return a * np.sin(b*x) + c

def interferometry():
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    image2 = np.array(plt.imread("Images/Interferometry10/25 020.bmp"), dtype=int)
    subtractionimage = np.abs(image1 - image2)
    summed_image = np.sum(subtractionimage, axis=0)
    params, params_cov = optimize.curve_fit(fit_function, np.arange(0, 744), summed_image, p0=[max(summed_image)-np.mean(summed_image), 1/40, np.mean(summed_image)])
    plt.scatter(np.arange(0, 744), summed_image)
    vals = np.arange(0, 744)
    plt.plot(np.arange(0, 744), fit_function(vals, params[0], params[1], params[2]))
    plt.title('Intensity values across image for 10° turn')
    plt.ylabel('Image Column Summed Value')
    plt.xlabel('Image x coordinate')
    plt.show()
    toimage(subtractionimage).show()


circumference = '6'
total_values = 739
xvalues = np.linspace(-0.5, 0.5, total_values)
deletion = 3
filepath = "Images/Images4/"
images = os.listdir(filepath)

# practicemain()
interferometry()
# widthlist()
