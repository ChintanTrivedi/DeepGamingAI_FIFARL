import numpy as np
import matplotlib.pyplot as plt


def moving_average_diff(a, n=200):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


r1 = np.loadtxt('model_epoch500/win_history.txt')
r2 = np.loadtxt('model_epoch1000/win_history.txt')
r3 = np.loadtxt('model_epoch1500/win_history.txt')

r2 = r2 + r1[-1]
r3 = r3 + r2[-1]

hist = np.concatenate((r1, r2, r3), axis=0)

plt.plot(moving_average_diff(hist))
# plt.plot([4 / ((x + 1) ** (1 / 2)) for x in range(800)])
plt.ylabel('Average #goals scored per attempt')
plt.xlabel('Epoch')
# plt.xlim([0, 900])
plt.title('Average #goals scored per attempt through 1000 training epochs')
# plt.ylim([0.2, 0.5])
plt.show()
