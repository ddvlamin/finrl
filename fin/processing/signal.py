import numpy as np

def days_to_hertz(days):
    days_in_year = 366
    sample_period = days_in_year * 24 * 3600  # in seconds = 1day
    sample_frequency = 1.0 / sample_period
    return (days_in_year / days) * sample_frequency


def filter_signal(signal, sample_frequency, cutoff_frequency=15):
    # Compute the FFT
    fft_result = np.fft.fft(signal)

    # Calculate the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(signal), 1.0 / sample_frequency)

    # plt.plot(frequencies, np.abs(fft_result), label='FFt')
    # plt.legend()
    # plt.show()

    fft_result[np.abs(frequencies) > cutoff_frequency] = 0

    # plt.plot(frequencies, np.abs(fft_result), label='FFt')
    # plt.legend()
    # plt.show()

    # Perform the inverse FFT
    filtered_signal = np.fft.ifft(fft_result)

    # Note: The result may contain small complex parts due to numerical precision.
    # You can convert it back to real numbers if needed.
    filtered_signal = np.real(filtered_signal)

    return filtered_signal


if __name__ == "__main__":
    import numpy as np

    fig, ax = plt.subplots()

    sample_period = 86400  # in seconds = 1day
    sample_frequency = 1.0 / sample_period
    time_frame_in_seconds = 5 * 366 * sample_period  # = five years in seconds
    samples = int(time_frame_in_seconds * sample_frequency)

    # Generate a sample signal (for demonstration purposes)
    t = np.linspace(0, time_frame_in_seconds, samples, endpoint=False)  # Time points
    frequency1 = (1 / 366) * sample_frequency  # Frequency of the first component (in Hz) = 1 year period
    frequency2 = (12 / 366) * sample_frequency  # Frequency of the second component (in Hz) = 1 month period
    frequency3 = (48 / 366) * sample_frequency  # Frequency of the second component (in Hz) = 1 week period
    signal = np.sin(2 * np.pi * frequency1 * t) + np.sin(2 * np.pi * frequency2 * t) + np.sin(
        2 * np.pi * frequency3 * t)
    filtered_signal = filter_signal(signal, sample_frequency, cutoff_frequency=(
                                                                                           24 / 366) * sample_frequency)  # everything with frequency faster than a 2 week period

    # Plot the original and filtered signals (optional)

    # ax.plot(t[:sample_frequency], signal[:sample_frequency], label='Original Signal')
    # ax.plot(t[:sample_frequency], filtered_signal[:sample_frequency], label='Filtered Signal')
    ax.plot(t, signal, label='Original Signal')
    ax.plot(t, filtered_signal, label='Filtered Signal')
    ax.legend()
    plt.show()