import numpy as np

# Given parameters
fc = 900e6  # Hz
alpha_g = -4  # dB
alpha_l = 10  # dB
path_loss = lambda d_km: 10 ** (0.1 * (120.9 + 37.6 * np.log10(d_km) + alpha_g + alpha_l))  # path loss in linear scale
noise_psd = 10 ** ((-174 + 5) / 10)  # noise power spectral density in linear scale (W/Hz)
distance = 0.002  # km
transmitted_power = 1e-3  # 1 mW

# Constants
c = 3e8  # speed of light

# Calculations
f_c = fc
d = distance * 1000  # convert distance to meters
g_t = 10 ** (alpha_g / 10)  # transmitter antenna gain in linear scale
g_r = 1  # assume receiver antenna gain of 1
l_p = path_loss(distance)
l_a = 10 ** (alpha_l / 10)  # antenna penetration loss in linear scale
l_pn = noise_psd * (f_c / 1e6)  # noise power in linear scale (W)
received_power = transmitted_power * g_t * g_r * l_p * l_a * l_pn / (
            4 * np.pi * (d ** 2) * f_c / c) ** 2  # received power in watts
channel_gain = received_power / transmitted_power  # channel gain in linear scale
normalized_channel_gain = channel_gain / noise_psd  # normalized channel gain with noise power of 1
normalized_channel_gain_dB = 10 * np.log10(normalized_channel_gain)  # normalized channel gain in dB

print(f'Normalized channel gain in dB: {normalized_channel_gain_dB:.2f} dB')
print(f'Normalized channel gain in linear Scale: {normalized_channel_gain:.2f}')
import numpy as np

# Assume the channel gain and noise power are given
channel_gain = np.array([0.5, 0.8, 1.2])  # channel gain for each channel
noise_power = 0.1  # noise power for each channel

# Normalize the channel gain to achieve a noise power of 1
normalized_gain = channel_gain / np.sqrt(noise_power)

print("Normalized channel gain:", normalized_gain)
