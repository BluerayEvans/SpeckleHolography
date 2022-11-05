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


def gaussian(x, a, b, c):
    return a * np.exp(-(((x - b) ** 2) / (2 * c ** 2)))


def halfwidth(fouriertransform):
    """returns the width of the peak"""
    halfmax = max(fouriertransform) / 2.35
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
    for y in images:
        imagearray = np.array(plt.imread(filepath + y), dtype=int)  # imports the image
        abs_transformed_column = transform_column(imagearray)
        width = halfwidth(abs_transformed_column)
        widths.append(width)
    x = np.arange(0, len(images))
    a, b = np.polyfit(x[1:-3], widths[1:-3], 1)
    widths = np.array(widths)
    plt.errorbar(x[1:-1] + 5, widths[1:-1], xerr=0.5, yerr=(widths[1:-1]*0.05), fmt='.k')
    plt.plot(x[1:-1] + 5, a * x[1:-1] + b, color='red')
    plt.title('Aperture circumference vs half width max')
    plt.xlabel('Aperture circumference (mm)')
    plt.ylabel('Half width max (k value)')
    plt.savefig('Plots/aperturecirchalfwidth.png', bbox_inches='tight')
    plt.show()
    return widths


def practicemain():
    """preliminary experiment"""
    num = 5
    imagearray = np.array(plt.imread(filepath + images[num]), dtype=int)  # imports the image
    abs_transformed_column = transform_column(imagearray)
    #params, pcov = optimize.curve_fit(gaussian, xvalues[int(deletion * 2 - 1):], abs_transformed_column, p0=[60000, 0, 0.1])
    plt.errorbar(xvalues[int(deletion * 2 - 1):], abs_transformed_column, fmt='.k')
    #plt.plot(xvalues[int(deletion * 2 - 1):], gaussian(xvalues[int(deletion * 2 - 1):], *params))
    plt.title('F.T. of the image data for circumference: ' + str(num+5) + 'mm')
    plt.xlabel('k value')
    plt.ylabel('Summed intensity along column')
    plt.savefig('Plots/FT' + str(num+5) + '.png', bbox_inches='tight')
    plt.show()


def sin_fit_function(x, a, b, c, d):
    return a * np.sin(b * x + d) + c


def normalisation_fit(summed_image):
    popt, pcov = optimize.curve_fit(pentic_fit_function, np.arange(0, 744), summed_image)
    return popt


def moving_average(window_size):
    """Calculates moving average of subtraction of 2 images for specified window size. Code based on Method 1 :
    https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/#:~:text=set%20of%20observations-,
    Method%201%3A%20Using%20Numpy,elements%20of%20the%20given%20array. """
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
    """function to test individual displacements"""
    window_size = 25
    moving_averages = moving_average(window_size)
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    deg = 50
    image2 = np.array(plt.imread("Images/Interferometry10/25 0" + str(deg) + ".bmp"), dtype=int)
    subtractionimage = np.abs(image1 - image2)
    summed_image = np.sum(subtractionimage, axis=0)
    xvals = np.arange(0, 744 - (window_size - 1))
    summed_image = summed_image[int(((window_size - 1) / 2)):-int(((window_size - 1) / 2))]
    difference = np.mean(summed_image) - moving_averages
    normalisedsummedimage = summed_image + difference
    plt.errorbar(xvals, normalisedsummedimage, fmt='.k')
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
    #plt.plot(xvals, sin_fit_function(xvals, *params))
    plt.title('Intensity values for ' + str(deg) + ' degree turn')
    plt.xlabel('Column of image')
    plt.ylabel('Summed intensity value')
    plt.savefig('Plots/intensityfringebias')
    plt.show()
    fft_summed_image = np.fft.fft(np.sum(subtractionimage, axis=0))
    frequency = abs(np.fft.fftfreq(744))
    plt.plot(frequency[12:], abs(fft_summed_image)[12:])
    plt.title('Fourier transform of summed fringes')
    plt.ylabel('Intensity correlation')
    plt.xlabel('Spatial frequency (px^-1)')
    plt.xlim(0, 0.05)
    plt.savefig('Plots/FToffringe.png')
    plt.show()
    # toimage(subtractionimage).show()


def linear(x, a, b):
    return a * x + b


def wavelengthpermeter():
    """function to calculate the displacement from the speckle and compare to the displacement from the screw"""
    window_size = 25  # how many values the moving average uses
    moving_averages = moving_average(window_size)  # finding the values to remove the general trend fluctuations
    xvals = np.arange(0, 744 - (window_size - 1))  # setting number of pixels accounting for the loss from moving avg
    numberoffringes = []
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)  # importing reference
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
        guessfrequency = 6 / fringelength
        params, params_cov = optimize.curve_fit(sin_fit_function, xvals, normalisedsummedimage,
                                                p0=[max(normalisedsummedimage) - np.mean(normalisedsummedimage),
                                                    guessfrequency,
                                                    np.mean(normalisedsummedimage), 0],
                                                sigma=((normalisedsummedimage / normalisedsummedimage) * 480))
        numberoffringes.append(744 / (6 / params[1]))
        error_on_wavenumber = np.sqrt(np.diag(params_cov))[1]
    meters_per_degree = 1.167 * 10 ** (-7)
    print(numberoffringes)
    wavelength = 780 * 10 ** (-9)
    laser_measured_displacement = np.array(numberoffringes) * wavelength * 0.75833333
    error_on_number_of_fringes = error_on_wavenumber * 744 / 6
    error_laser_measured_displacement = error_on_number_of_fringes * wavelength * 0.2
    displacement = degrees * meters_per_degree
    error_on_y = displacement * 0.2
    popt, pcov = optimize.curve_fit(linear, laser_measured_displacement, displacement, sigma=error_on_y)
    print(popt)
    print(pcov)
    plt.errorbar(laser_measured_displacement, displacement, xerr=error_laser_measured_displacement, yerr=error_on_y,
                 fmt='.k')
    plt.plot([0, 9*10**(-6)], [0, 9*10**(-6)], c='red', label='gradient 1')
    plt.plot(laser_measured_displacement, linear(laser_measured_displacement, *popt), label='Fit to data')
    plt.legend(loc='upper left')
    plt.title('Laser measured displacement vs screw measured displacement')
    plt.xlabel('Laser calculated displacement (m)')
    plt.ylabel('Screw measured displacement (m)')

    plt.savefig('Plots/fixedfinal')
    plt.show()


circumference = '6'
total_values = 739
xvalues = np.linspace(-0.5, 0.5, total_values)
deletion = 3
filepath = "Images/Images4/"
images = os.listdir(filepath)

#practicemain()
interferometry()
wavelengthpermeter()
#widthlist()
