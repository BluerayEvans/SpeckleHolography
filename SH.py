import os
import numpy as np
import matplotlib.pyplot as plt
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


def pentic_fit_function(x, a, b, c, d, e, f, g, h, i):
    return a * (x - b)** 4 + c * (x - d) ** 3 + e * (x - f) ** 2 + g * (x - h) + i


def normalisation_fit(summed_image):
    popt, pcov = optimize.curve_fit(pentic_fit_function, np.arange(0, 744), summed_image)
    return popt


'''def findfringes(normalisedsummedimage, fringelength):
    params, params_cov = optimize.curve_fit(fit_function, np.arange(0, 744), normalisedsummedimage,
                                            p0=[max(normalisedsummedimage) - np.mean(normalisedsummedimage), fringelength,
                                                np.mean(normalisedsummedimage)])
    
    plt.plot(np.arange(0, 744), fit_function(vals, params[0], params[1], params[2]))
    return 1/params[1]'''

def interferometry():
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    image2 = np.array(plt.imread("Images/Interferometry10/25 090.bmp"), dtype=int)
    subtractionimage = np.abs(image1 - image2)
    summed_image = np.sum(subtractionimage, axis=0)
    # popt = normalisation_fit(summed_image)
    plt.scatter(np.arange(0, 744), summed_image)
    xvals = np.arange(0, 744)
    popt, pcov = optimize.curve_fit(pentic_fit_function, np.arange(0, 744), summed_image)
    # plt.plot(xvals, pentic(xvals, popt[0], popt[1], popt[2], popt[3], popt[4]))
    normalisedsummedimage = summed_image
    fringelength = 0.0067*6
    params, params_cov = optimize.curve_fit(sin_fit_function, np.arange(0, 744), normalisedsummedimage,
                                            p0=[max(normalisedsummedimage) - np.mean(normalisedsummedimage),
                                                fringelength,
                                                np.mean(normalisedsummedimage), 0])
    print(params[1])
    plt.plot(xvals, sin_fit_function(xvals, params[0], params[1], params[2], params[3]))
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


def wavelengthpermeter():
    numberoffringes = []
    image1 = np.array(plt.imread("Images/Interferometry10/25 000.bmp"), dtype=int)
    frequency = np.abs(np.fft.fftfreq(744))
    degrees = np.arange(1, 9) * 10
    for x in degrees:
        image2 = np.array(plt.imread("Images/Interferometry10/25 0" + str(x) + ".bmp"), dtype=int)
        subtractionimage = np.abs(image1 - image2)
        summed_image = np.sum(subtractionimage, axis=0)
        fft_summed_image = np.abs(np.fft.fft(summed_image))
        fringelength = 1 / ((frequency[int(x / 10):372])[np.argmax(np.abs(fft_summed_image[int(x / 10):372]))])
        print(fringelength)
        normalisedsummedimage = summed_image
        print(1/fringelength)
        guessfrequency = 6 * 1/fringelength
        params, params_cov = optimize.curve_fit(sin_fit_function, np.arange(0, 744), normalisedsummedimage,
                                                p0=[max(normalisedsummedimage) - np.mean(normalisedsummedimage),
                                                    guessfrequency,
                                                    np.mean(normalisedsummedimage), 0])

        numberoffringes.append(744 / (6/params[1]))
    meters_per_degree = 1.167 * 10 ** (-7)
    print(numberoffringes)
    displacement = degrees * meters_per_degree
    plt.plot(displacement, numberoffringes)
    plt.show()


circumference = '6'
total_values = 739
xvalues = np.linspace(-0.5, 0.5, total_values)
deletion = 3
filepath = "Images/Images4/"
images = os.listdir(filepath)

# practicemain()
interferometry()
#wavelengthpermeter()
# widthlist()
