import numpy as np
import matplotlib.pyplot as plt
import torch

# Parameters
N = 1000        # length of signal
fs = 50         # sample rate
t = np.arange(N) / fs  # time vector
# Wavelet parameters
scale_min = 1
scale_max = 128
num_scales = 100
scales = np.linspace(scale_min, scale_max, num_scales)


def chrip():
    # Chirp signal parameters
    f0 = 1           # starting frequency
    f1 = 10          # ending frequency
    t1 = N / fs      # end time
    t_half = t1 / 2  # midpoint time

    # Generate chirp signal
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t1)))

def conv(sig, psi):
    psi_len = len(psi)
    sig_len = len(sig)
    psi = np.conj(psi)
    zero_padding = np.zeros(psi_len - 1,)
    sig = np.concatenate((zero_padding, sig, zero_padding))
    out = np.empty_like(sig)
    for i in range(sig_len):
        out[i] = sig[i:i + psi_len] * psi
    
    return out

def morlet2(t, scale):
    return np.pi**(-0.25) * np.exp(1j * 2 * np.pi * t) * np.exp(-t**2 / 2 / scale**2)

def morlet(t):
    return np.exp(-t**2/2) * np.cos(5*t)

########################################################################################
# Initialize coefficients matrix
coeffs = np.zeros((num_scales, N), dtype=complex)

# Compute CWT
for i, scale in enumerate(scales):
    # Scaled wavelet
    # psi_scale = morlet_wavelet(t * scale, scale)
    # plt.plot(psi_scale)
    # plt.plot(np.real(chrip()))
    # Convolution
    # coeffs[i, :] = np.convolve(chrip(), psi_scale, mode='same')

    sig = chrip()
    psi = morlet(np.linspace(-100,100,100))
    plt.clf()
    plt.plot(sig)
    plt.plot(psi)
    plt.show()
    # conv(sig, psi)

# Plot the scalogram
plt.figure(figsize=(10, 6))
# plt.imshow(np.abs(coeffs), extent=[0, t[-1], scale_max, scale_min], aspect='auto', cmap='jet')
plt.imshow(np.abs(coeffs),  aspect='auto', cmap='jet')
plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.title('Scalogram of Chirp Signal')
plt.show()
