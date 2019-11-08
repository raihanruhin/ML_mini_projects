from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = [1, 2, 3, 4, 5]
ys = [5, 4, 6, 5, 6]

#plt.scatter(xs, ys)
#plt.show()
xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

def best_fit_slop(xs, ys):
	m = (mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
	return m

m = best_fit_slop(xs, ys)
print(m)