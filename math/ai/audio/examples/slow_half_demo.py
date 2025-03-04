import time, os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Generate "Hello World" audio (macOS version)
# os.system("say -o hello_world.aiff --file-format=AIFF --data-format=LEI16@44100 'Hello World'")

# Convert AIFF to WAV using pydub
# sound = AudioSegment.from_file("hello_world.aiff", format="aiff")
# sound.export("hello_world.wav", format="wav")

# Load WAV file
sr_original, data_original = wav.read("hello_world.wav")  # sr = sample rate, data = audio samples
sr_good_stretch, data_good_stretch = wav.read("hello_world_good_stretch.wav")  # sr = sample rate, data = audio samples
sr_bad_stretch, data_bad_stretch = wav.read("hello_world_bad_stretch.wav")  # sr = sample rate, data = audio samples

print(len(data_original))
print(len(data_good_stretch))
print(len(data_bad_stretch))

print(sr_original)
print(sr_good_stretch)
print(sr_bad_stretch)

# Function to plot waveforms
def plot_waveforms(data_original, data_bad_stretch, data_good_stretch, 
               sr_original, sr_bad_stretch, sr_good_stretch):
    time_orig = np.linspace(0, len(data_original) / sr_original, num=len(data_original))  # Time axis for original
    time_bad_stretch = np.linspace(0, len(data_bad_stretch) / sr_bad_stretch, num=len(data_bad_stretch))  # Time axis for original
    time_good_stretch = np.linspace(0, len(data_good_stretch) / sr_good_stretch, num=len(data_good_stretch))  # Time axis for original

    # Downsample for plotting clarity (reduce points)
    downsample_factor = 100  
    data_original_down = data_original[::downsample_factor]
    data_bad_stretch_down = data_bad_stretch[::downsample_factor]
    data_good_stretch_down = data_good_stretch[::downsample_factor]
    time_orig_down = time_orig[::downsample_factor]
    time_bad_stretch_down = time_bad_stretch[::downsample_factor]
    time_good_stretch_down = time_good_stretch[::downsample_factor]

    # Compute Fourier
    cutoff_freq = 4000
    fft_orig = np.fft.fft(data_original.astype(float))
    freqs_orig = np.fft.fftfreq(len(data_original), d=1/sr_original)
    fft_orig_magnitude = np.abs(fft_orig)
    pos_idx_orig = np.bitwise_and(freqs_orig >= 0, freqs_orig < cutoff_freq)

    fft_bad_stretch = np.fft.fft(data_bad_stretch.astype(float))
    freqs_bad_stretch = np.fft.fftfreq(len(data_bad_stretch), d=1/sr_bad_stretch)
    fft_bad_stretch_magnitude = np.abs(fft_bad_stretch)
    pos_idx_bad_stretch = np.bitwise_and(freqs_bad_stretch >= 0, freqs_bad_stretch < cutoff_freq)

    fft_good_stretch = np.fft.fft(data_good_stretch.astype(float))
    freqs_good_stretch = np.fft.fftfreq(len(data_good_stretch), d=1/sr_good_stretch)
    fft_good_stretch_magnitude = np.abs(fft_good_stretch)
    pos_idx_good_stretch = np.bitwise_and(freqs_good_stretch >= 0, freqs_good_stretch < cutoff_freq) 

    # Plot both waveforms
    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex='col')
    
    axs[0, 0].plot(time_orig_down, data_original_down, color='b')
    axs[0, 0].set_title("'Hello World' Original Sound Wave")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].set_xlabel("Time (s)")

    axs[0, 1].plot(freqs_orig[pos_idx_orig], fft_orig_magnitude[pos_idx_orig], color='green')
    axs[0, 1].set_title("Original Audio Spectrum (Frequency Domain)")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Magnitude")

    axs[1, 0].plot(time_bad_stretch_down, data_bad_stretch_down, color='b')
    axs[1, 0].set_title("'Hello World' Bad Stretched Sound Wave")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].set_xlabel("Time (s)")

    axs[1, 1].plot(freqs_bad_stretch[pos_idx_bad_stretch], fft_bad_stretch_magnitude[pos_idx_bad_stretch], color='green')
    axs[1, 1].set_title("Bad Audio Spectrum (Frequency Domain)")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Magnitude")

    axs[2, 0].plot(time_good_stretch_down, data_good_stretch_down, color='b')
    axs[2, 0].set_title("'Hello World' Good Stretched Sound Wave")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Amplitude")

    axs[2, 1].plot(freqs_good_stretch[pos_idx_good_stretch], fft_good_stretch_magnitude[pos_idx_good_stretch], color='green')
    axs[2, 1].set_title("Good Audio Spectrum (Frequency Domain)")
    axs[2, 1].set_xlabel("Frequency (Hz)")
    axs[2, 1].set_ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

# Reload slowed-down file and play
# sd.play(data_original, sr_original)
# sd.wait()
# time.sleep(1)
# sd.play(data_bad_stretch, sr_bad_stretch)
# sd.wait()
# time.sleep(1)
# sd.play(data_good_stretch, sr_good_stretch)
# sd.wait()
# time.sleep(1)

plot_waveforms(data_original, data_bad_stretch, data_good_stretch, 
               sr_original, sr_bad_stretch, sr_good_stretch)
