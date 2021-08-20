import numpy as np
import matplotlib.pyplot as plt

def plot(spread):
    fig1, ax1 = plt.subplots()
    plt.scatter([1] * len(spread), spread)
    ax1.set_title('Basic Plot')
    ax1.boxplot(spread)
    plt.show()

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
spread = np.random.rand(50) * 100
plot(spread)

