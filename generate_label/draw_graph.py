import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 6)
y = x * x

plt.plot(x, y, marker='o')
last_xy = (0,0)

for xy in zip(x, y):
	
    plt.annotate("w = 100", xy=((xy[0]+last_xy[0])/2,(xy[1]+last_xy[1])/2), xytext=(0,0), textcoords='offset points')
    last_xy = xy
plt.show()