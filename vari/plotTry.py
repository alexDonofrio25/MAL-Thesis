import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a = [2,9]
print(a)
a.append(0.5)
print(a)
x = [0.5,0.1,0.8]
y = [0.3,0.7,0.05]
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.plot(x, y, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="white")
plt.show()

for i in [0,5]:
    x = [i*0.1]
    y = [i*0.2]
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.plot(x, y, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="white")
    plt.show()
