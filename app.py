import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert

# === Streamlit UI ===
st.title("Bearing Fault Vibration Signal Simulator")

st.sidebar.header("Bearing Geometry Parameters")
n = st.sidebar.number_input("Number of Rolling Elements (n)", value=7)
R_mm = st.sidebar.number_input("Pitch Diameter (R) [mm]", value=60.0)
d_mm = st.sidebar.number_input("Ball Diameter (d) [mm]", value=20.0)
beta_deg = st.sidebar.number_input("Contact Angle (β) [deg]", value=0.0)
fs = st.sidebar.number_input("Sampling Frequency (fs) [Hz]", value=12000)
duration = st.sidebar.number_input("Signal Duration [s]", value=2.0)
RPM = st.sidebar.number_input("Shaft Speed (RPM)", value=1800)
load_variation = st.sidebar.slider("Load-Induced Modulation", 0.0, 1.0, 0.2)

fault_type = st.selectbox("Fault Type", ["outer", "inner", "ball"])
fault_size_mm = st.number_input("Fault Size [mm]", value=0.5334)
fault_depth = st.slider("Fault Depth (for future use)", 0.0, 1.0, 0.5)

# === Convert Units ===
R = R_mm / 1000  # [m]
d = d_mm / 1000  # [m]
beta = np.deg2rad(beta_deg)

# === Derived Time Vector ===
Omega = RPM * 2 * np.pi / 60
t = np.linspace(0, duration, int(fs * duration))

# === Fault Size Mappings ===
def size_to_amplitude(size_mm):
    min_size, max_size = 0.1778, 1.02
    min_amp, max_amp = 0.5, 2.0
    return ((size_mm - min_size) / (max_size - min_size)) * (max_amp - min_amp) + min_amp

def size_to_spikewidth(size_mm):
    min_size, max_size = 0.1778, 1.02
    min_width, max_width = 0.0005, 0.002
    return ((size_mm - min_size) / (max_size - min_size)) * (max_width - min_width) + min_width

# === Bearing Frequencies ===
def bearing_fault_frequencies(n, d, R, beta, RPM):
    fr = RPM / 60
    FTF = 0.5 * fr * (1 - (d / R) * np.cos(beta))
    BPFI = 0.5 * n * fr * (1 + (d / R) * np.cos(beta))
    BPFO = 0.5 * n * fr * (1 - (d / R) * np.cos(beta))
    BSF = R / d * fr * (1 - ((d / R * np.cos(beta)) ** 2))
    return FTF, BPFI, BPFO, BSF

FTF, BPFI, BPFO, BSF = bearing_fault_frequencies(n, d, R, beta, RPM)

if fault_type == 'outer':
    f_fault = BPFO
elif fault_type == 'inner':
    f_fault = BPFI
elif fault_type == 'ball':
    f_fault = BSF
else:
    f_fault = BPFO

st.write(f"**BPFO**: {BPFO:.2f} Hz | **BPFI**: {BPFI:.2f} Hz | **BSF**: {BSF:.2f} Hz | **FTF**: {FTF:.2f} Hz")

# === Generate Fault Signal ===
def generate_fault_impulses(fault_freq, fs, duration, fault_size_mm):
    amplitude = size_to_amplitude(fault_size_mm)
    spike_width = size_to_spikewidth(fault_size_mm)
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    period = 1 / fault_freq
    num_cycles = int(duration / period)
    spike_samples = int(spike_width * fs)

    for i in range(num_cycles):
        center = int(i * period * fs)
        if center + spike_samples < len(t):
            spike_range = np.arange(center - spike_samples // 2, center + spike_samples // 2)
            spike_range = spike_range[(spike_range >= 0) & (spike_range < len(t))]
            spike_time = t[spike_range] - t[center]
            impulse = amplitude * np.exp(-((spike_time * fs) ** 2) / 2)
            signal[spike_range] += impulse
    return signal

raw_fault_signal = generate_fault_impulses(f_fault, fs, duration, fault_size_mm)
shaft_signal = 0.3 * np.sin(2 * np.pi * (RPM / 60) * t)
modulated_fault_signal = raw_fault_signal * (1 + shaft_signal)

# === Bandpass Filter ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

filtered_signal = bandpass_filter(modulated_fault_signal, 2000, 5500, fs)

# === Add Noise ===
load_modulation = 1 + load_variation * np.sin(2 * np.pi * 0.5 * t)
noise = 0.05 * np.random.randn(len(t))
vibration_signal = filtered_signal * load_modulation + noise

# === Envelope Detection & FFT ===
analytic_signal = hilbert(vibration_signal)
envelope = np.abs(analytic_signal)

N = len(t)
freqs = np.fft.rfftfreq(N, 1 / fs)
fft_spectrum = np.abs(np.fft.rfft(vibration_signal))
envelope_spectrum = np.abs(np.fft.rfft(envelope))

# === Plotting with Matplotlib ===
fig, axs = plt.subplots(3, 1, figsize=(15, 10))

axs[0].plot(t[:2000], vibration_signal[:2000], label="Time-domain Signal")
axs[0].axhline(0, color='black', linewidth=0.5)
axs[0].set_title(f"{fault_type.capitalize()} Fault Vibration Signal - Fault Size {fault_size_mm} mm")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True)

def plot_fault_harmonics(ax, freq, max_freq):
    h = freq
    while h < max_freq:
        ax.axvline(h, color='red', linestyle='--', linewidth=1)
        h += freq

axs[1].plot(freqs, fft_spectrum, label="FFT Spectrum")
plot_fault_harmonics(axs[1], f_fault, fs / 2)
axs[1].set_title("Frequency Spectrum")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(freqs, envelope_spectrum, color='green', label="Envelope Spectrum")
plot_fault_harmonics(axs[2], f_fault, fs / 2)
axs[2].set_title("Envelope Spectrum")
axs[2].set_xlabel("Frequency [Hz]")
axs[2].set_ylabel("Amplitude")
axs[2].legend()
axs[2].grid(True)

fig.tight_layout()
st.pyplot(fig)
