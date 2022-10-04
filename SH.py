import numpy as np
import matplotlib.pyplot as plt


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
    for x in range(5, 14):
        imagearray = plt.imread("Images/Images2/" + str(x) + "mm circ speckle image.bmp")  # imports the image
        column = imagearray[240, :]
        transformed_column = np.fft.fft(column)
        transformed_column = np.concatenate((transformed_column[372:], transformed_column[1:372]))
        abs_transformed_column = abs(transformed_column) ** 0.5
        width = halfwidth(abs_transformed_column)
        widths.append(width)

    plt.scatter([6, 7, 8, 9, 10, 11, 12], widths[1:-1])

    plt.title('Aperture circumference vs half width max of fourier transform')
    plt.xlabel('Aperture circumference (mm)')
    plt.ylabel('Half width max (mm)')
    plt.show()
    return widths


def main():
    imagearray = plt.imread("Images/Images2/" + circumference + "mm circ speckle image.bmp")  # imports the image
    print(imagearray.shape)
    column = imagearray[240, :]
    print(column)

    transformed_column = np.fft.fft(column)
    print(transformed_column)
    transformed_column = np.concatenate((transformed_column[372:], transformed_column[1:372]))
    abs_transformed_column = abs(transformed_column)
    print(halfwidth(abs_transformed_column))
    plt.plot(xvalues[1:], abs_transformed_column)
    plt.title('Absolute values from fourier transform of 1 column of the image data for circumference: ' + circumference)
    plt.xlabel('Sensor width (mm)')
    plt.ylabel('Amplitude')
    plt.savefig('Plots/absolute.png', bbox_inches='tight')
    plt.show()
    print(halfwidth(transformed_column * np.conjugate(transformed_column)))
    plt.plot(xvalues[1:], abs(transformed_column * np.conjugate(transformed_column)))
    plt.title('Squared values from fourier transform of 1 column of the image data for circumference: ' + circumference)
    plt.xlabel('Sensor width (mm)')
    plt.ylabel('Amplitude')
    plt.savefig('Plots/squared.png', bbox_inches='tight')
    plt.show()
    print(halfwidth(abs(transformed_column ** 0.5)))
    plt.plot(xvalues[1:], abs(transformed_column ** 0.5))
    plt.title('Rooted values from fourier transform of 1 column of the image data for circumference: ' + circumference)
    plt.xlabel('Sensor width (mm)')
    plt.ylabel('Amplitude')
    plt.savefig('Plots/rooted.png', bbox_inches='tight')
    plt.show()


circumference = '13'
xvalues = np.linspace(0, 2.97, 744)
main()
print(widthlist())
