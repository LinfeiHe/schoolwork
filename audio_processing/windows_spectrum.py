import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

WINDOW_SIZE = 63
WINDOW_TYPE = 'blackmanharris'
FFT_SIZE = 256

# get window
window = signal.get_window(WINDOW_TYPE, WINDOW_SIZE)

# zero-padding
hw1 = int(np.floor((WINDOW_SIZE+1)/2))
hw2 = int(np.floor(WINDOW_SIZE/2))
hf = FFT_SIZE / 2
temp = np.zeros(FFT_SIZE, np.float)
temp[:hw1] = window[hw2:]
temp[FFT_SIZE - hw2:] = window[:hw2]

# FFT
x = np.fft.fft(temp)
abs_x = abs(x)
abs_x[abs_x < np.finfo(float).eps] = np.finfo(float).eps
m_x = 20 * np.log10(abs_x)
p_x = np.angle(x)

m_x1 = np.zeros(FFT_SIZE)
p_x1 = np.zeros(FFT_SIZE)
m_x1[:hf] = m_x[hf:]
m_x1[FFT_SIZE-hf:] = m_x[:hf]
p_x1[:hf] = p_x[hf:]
p_x1[FFT_SIZE-hf:] = p_x[:hf]


# draw
plt.subplot(311)
plt.plot(np.arange(-hw1, hw2), window)
plt.axis([-hw1, hw2, 0, 1.2])
string = WINDOW_TYPE + '\n' + 'M=%d' %WINDOW_SIZE + '  N=%d' %FFT_SIZE
plt.title(string)

plt.subplot(312)
plt.plot(np.arange(-hf, hf), m_x1-max(m_x1))
plt.axis([-hf, hf, -200, 0])

plt.subplot(313)
plt.plot(np.arange(-hf, hf), p_x1)
plt.axis([-hf, hf, -5, 5])


plt.show()
