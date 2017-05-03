import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

M = 200

# generate triangle wave
data = []
for i in range(0, 10):
    temp = np.linspace(0, 1, 20)
    data = np.r_[data, temp]

# fft transform
theta = np.angle(np.fft.fft(data))
data_fft = abs(np.fft.fft(data))

# add window
window = signal.hann(M)
data_win = window * data[:M]
theta_win = np.angle(np.fft.fft(data_win))
data_win_fft = abs(np.fft.fft(data_win))


# draw
plt.subplot(321)
plt.plot(data)
plt.title('Triangle wave')
plt.subplot(323)
plt.plot(data_fft)
plt.subplot(325)
plt.plot(theta)

plt.subplot(322)
plt.plot(data_win)
plt.title('Add hanning window')
plt.subplot(324)
plt.plot(data_win_fft)
plt.subplot(326)
plt.plot(theta_win)

plt.show()

