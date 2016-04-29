import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import GaussianAbsorption1D

plt.figure()
s1 = GaussianAbsorption1D()
r = np.arange(-5, 5, .01)
for factor in range(1, 4):
    s1.amplitude = factor
    plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

plt.axis([-5, 5, -3, 2])
plt.show()