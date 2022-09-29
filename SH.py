import numpy as np
import matplotlib.pyplot as plt


def halfwidth(fouriertransform):
    """returns the width of the peak"""
    halfmax = max(fouriertransform) / 2
    for x in range(0, len(fouriertransform) - 1):
        if fouriertransform[x] >= halfmax:
            left = x
            break
    for x in range(len(fouriertransform) - 1, 0, -1):
        if fouriertransform[x] >= halfmax:
            right = x
            break
    return right - left


imagearray = plt.imread("Images/6mm circ speckle image.bmp")  # imports the image
print(imagearray.shape)
column = imagearray[240, :]
print(column)

transformed_column = np.fft.fft(column)
print(transformed_column)
transformed_column = np.concatenate((transformed_column[372:], transformed_column[1:372]))
xvalues = np.arange(0, 744)
abs_transformed_column = abs(transformed_column)
print(halfwidth(abs_transformed_column))
plt.plot(xvalues[1:], abs_transformed_column)
plt.savefig('Plots/absolute.png', bbox_inches='tight')
plt.show()
print(halfwidth(transformed_column * np.conjugate(transformed_column)))
plt.plot(xvalues[1:], abs(transformed_column * np.conjugate(transformed_column)))
plt.savefig('Plots/squared.png', bbox_inches='tight')
plt.show()
print(halfwidth(abs(transformed_column ** 0.5)))
plt.plot(xvalues[1:], abs(transformed_column ** 0.5))
plt.savefig('Plots/rooted.png', bbox_inches='tight')
plt.show()
