import numpy as np
import matplotlib.pyplot as plt


samples = [np.random.beta(100, 900) for i in range(10000)]
plt.hist(samples, bins='auto')
plt.show()