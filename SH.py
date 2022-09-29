import numpy as np
import matplotlib.pyplot as plt

imagearray = plt.imread("Images/6mm circ speckle image.bmp")
print(imagearray.shape)
column = imagearray[240, :]
print(column)

transformed_column = np.fft.fft(column)
print(transformed_column)
transformed_column = np.concatenate((transformed_column[372:], transformed_column[1:372]))
xvalues = np.arange(0, 744)
plt.plot(xvalues[1:], abs(transformed_column))
plt.savefig('Plots/absolute.png', bbox_inches='tight')
plt.show()
plt.plot(xvalues[1:], transformed_column*np.conjugate(transformed_column))
plt.savefig('Plots/squared.png', bbox_inches='tight')
plt.show()
plt.plot(xvalues[1:], (transformed_column**0.5))
plt.savefig('Plots/rooted.png', bbox_inches='tight')
plt.show()
