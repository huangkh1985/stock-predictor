# src/ml/data/preprocessing.py

import pywt
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------
# Wavelet denoising
#
# Reference: https://doi.org/10.1002/for.3071
# (the idea of the wavelet denoising and some parameters are taken from this paper)
# -------------------------------------------------------------------------------------------

# Function for padding the data
def pad_data(data, pad_width, mode='edge'):
    return np.pad(data, pad_width, mode=mode)

# Wavelet denoising function with parameterisation for wavelet type and decomposition level
def wavelet_denoising(data, wavelet='db4', level=1):
    # Padding with a width of 100
    padded_data = pad_data(data, pad_width=100, mode='edge')
    # Decompose signal using Wavelet Transform
    coeff = pywt.wavedec(padded_data, wavelet, mode="per", level=level)
    # Estimate the noise level
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    # Calculate the universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(padded_data)))
    # Apply soft thresholding to detail coefficients
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    # Set high-frequency coefficients to zero
    coeff[-level] = np.zeros_like(coeff[-level])
    # Reconstruct the denoised signal
    denoised_data = pywt.waverec(coeff, wavelet, mode='per')
    # Remove the padding
    denoised_data = denoised_data[100:-100]  # Adjust this if necessary

    # Handle edge effects
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:len(data)]
    elif len(denoised_data) < len(data):
        denoised_data = np.pad(denoised_data, (0, len(data) - len(denoised_data)), 'edge')

    return denoised_data

def plot_wavelet_denoising(df, stock):
    # Print some metrics for verification
    snr = 10 * np.log10(np.sum(df['Close_denoised'] ** 2) / np.sum(df['Noise'] ** 2))
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

    # Plotting the results
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
    loc = 'upper left'

    # Plot Original Signal
    axes[0].plot(df.index, df['Close'], label='Original Signal', color='blue')
    axes[0].legend(loc=loc)
    axes[0].grid(False)  # Disable grid

    # Plot Denoised Signal
    axes[1].plot(df.index, df['Close_denoised'], label='Denoised Signal', color='green')
    axes[1].legend(loc=loc)
    axes[1].grid(False)  # Disable grid

    # Plot Noise
    axes[2].plot(df.index, df['Noise'], label='Extracted Noise', color='red')
    axes[2].legend(loc=loc)
    axes[2].grid(False)  # Disable grid

    fig.suptitle(f"Wavelet Denoising for {stock}", fontsize=16)

    plt.tight_layout() 
    plt.show()

# -------------------------------------------------------------------------------------------
# Process dates
# -------------------------------------------------------------------------------------------

def process_dates(df):
    # Convert the Date column to time zone-naive datetime
    return df.tz_localize(None)
