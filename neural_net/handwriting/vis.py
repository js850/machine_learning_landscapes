import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# grid = np.loadtxt("X2.dat")
dat = np.loadtxt("X.dat")

x = dat[1000]
grid = x.reshape((np.sqrt(len(x)),np.sqrt(len(x))))

# plt.imshow(grid,interpolation='nearest')
plt.imshow(grid,interpolation='nearest',cmap=plt.gray())
plt.show()