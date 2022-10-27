import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import optimize
import pandas as pd


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
    a, b = np.polyfit(np.delete(x, [1, 6, 7, 8]), np.delete(widths, [1, 6, 7, 8]), 1)
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


def sin_fit_function(x, a, b, c, d):
    return a * np.sin(b * x + d) + c


def normalisation_fit(summed_image):
    popt, pcov = optimize.curve_fit(pentic_fit_function, np.arange(0, 744), summed_image)
    return popt


def moving_average(window_size):
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    image2 = np.array(plt.imread("Images/Interferometry10/25 090.bmp"), dtype=int)
    subtractionimage = np.abs(image1 - image2)
    summed_image = np.sum(subtractionimage, axis=0)
    arr = summed_image
    i = 0
    moving_averages = []
    while i < len(arr) - window_size + 1:
        window_average = round(np.sum(arr[
                                      i:i + window_size]) / window_size, 2)
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def interferometry():
    window_size = 25
    moving_averages = moving_average(window_size)
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    deg = 70
    image2 = np.array(plt.imread("Images/Interferometry10/25 0" + str(deg) + ".bmp"), dtype=int)
    subtractionimage = np.abs(image1 - image2)
    summed_image = np.sum(subtractionimage, axis=0)
    xvals = np.arange(0, 744-(window_size-1))
    summed_image = summed_image[int(((window_size-1)/2)):-int(((window_size-1)/2))]
    difference = np.mean(summed_image) - moving_averages
    normalisedsummedimage = summed_image + difference
    plt.scatter(xvals, normalisedsummedimage)
    fft_summed_image = np.abs(np.fft.fft(normalisedsummedimage))
    frequency = np.abs(np.fft.fftfreq(744))
    fringelength = 1 / ((frequency[int(deg / 10):372])[np.argmax(np.abs(fft_summed_image[int(deg / 10):372]))])
    print(fringelength)
    print(1 / fringelength)
    guessfrequency = 6 * 1 / fringelength
    params, params_cov = optimize.curve_fit(sin_fit_function, xvals, normalisedsummedimage,
                                            p0=[max(normalisedsummedimage) - np.mean(normalisedsummedimage),
                                                guessfrequency,
                                                np.mean(normalisedsummedimage), 0])
    print(params[1])
    plt.plot(xvals, sin_fit_function(xvals, *params))
    plt.show()
    '''fft_summed_image = np.fft.fft(summed_image)
    frequency = abs(np.fft.fftfreq(744))
    plt.plot(frequency[12:], abs(fft_summed_image)[12:])
    plt.title('Fourier transform of summed fringes')
    plt.ylabel('Intensity correlation')
    plt.xlabel('Spatial frequency (px^-1)')
    plt.xlim(0, 0.05)
    plt.show()'''
    # toimage(subtractionimage).show()


def linear(x, a, b):
    return a*x+b


def wavelengthpermeter():
    window_size = 9
    moving_averages = moving_average(window_size)
    xvals = np.arange(0, 744-(window_size-1))
    numberoffringes = []
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    frequency = np.abs(np.fft.fftfreq(744))
    degrees = np.arange(1, 9) * 10
    for x in degrees:
        image2 = np.array(plt.imread("Images/Interferometry10/25 0" + str(x) + ".bmp"), dtype=int)
        subtractionimage = np.abs(image1 - image2)
        summed_image = np.sum(subtractionimage, axis=0)
        summed_image = summed_image[int(((window_size - 1) / 2)):-int(((window_size - 1) / 2))]
        fft_summed_image = np.abs(np.fft.fft(summed_image))
        fringelength = 1 / ((frequency[int(x / 10):372])[np.argmax(np.abs(fft_summed_image[int(x / 10):372]))])
        difference = np.mean(summed_image) - moving_averages
        normalisedsummedimage = summed_image + difference
        guessfrequency = 6 * 1/fringelength
        params, params_cov = optimize.curve_fit(sin_fit_function, xvals, normalisedsummedimage,
                                                p0=[max(normalisedsummedimage) - np.mean(normalisedsummedimage),
                                                    guessfrequency,
                                                    np.mean(normalisedsummedimage), 0])
        numberoffringes.append(744 / (6/params[1]))
    meters_per_degree = 1.167 * 10 ** (-7)
    print(numberoffringes)
    numberoffringes = np.array(numberoffringes) * 780*10**(-9) * 0.7
    displacement = degrees * meters_per_degree
    plt.scatter(numberoffringes, displacement)
    popt, pcov = optimize.curve_fit(linear, numberoffringes, displacement)
    print(popt)
    plt.plot(numberoffringes, linear(numberoffringes, *popt))
    plt.show()


circumference = '6'
total_values = 739
xvalues = np.linspace(-0.5, 0.5, total_values)
deletion = 3
filepath = "Images/Images4/"
images = os.listdir(filepath)

# practicemain()
#interferometry()
wavelengthpermeter()
# widthlist()
