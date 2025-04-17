import streamlit as st
import numpy as np
from scipy.signal import butter, lfilter, hilbert
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Span

# === Sidebar Inputs: Bearing Geometry Parameters ===
st.sidebar.header("Bearing Geometry Parameters")
n = st.sidebar.number_input("Number of Rolling Elements (n)", value=7)
R = st.sidebar.number_input("Pitch Diameter [mm]", value=60.0) / 1000  # Convert to m
d = st.sidebar.number_input("Ball Diameter [mm]", value=20.0) / 1000   # Convert to m
beta = np.deg2rad(st.sidebar.slider("Contact Angle [°]", min_value=0, max_value=30, value=0))
fs = st.sidebar.number_input("Sampling Frequency [Hz]", value=12000)
duration = st.sidebar.number_input("Signal Duration [s]", value=2.0)
RPM = st.sidebar.number_input("Shaft Speed [RPM]", value=1800)
load_variation = st.sidebar.slider("Load Variation Multiplier", 0.0, 1.0, 0.2)

# === Main Page Inputs: Fault Type and Size ===
st.title("Synthetic Bearing Fault Signal Simulator")
fault_type = st.selectbox("Select Fault Type", options=['outer', 'inner', 'ball'])
fault_size_mm = st.number_input("Fault Size [mm]", value=0.5334, min_value=0.1)
fault_depth = st.number_input("Fault Depth (for visualization only)", value=1.0)

# === Time and Derived Params ===
Omega = RPM * 2 * np.pi / 60
t = np.linspace(0, duration, int(fs * duration))

# === Mapping Functions ===
def size_to_amplitude(size_mm):
    min_size, max_size = 0.1778, 1.02
    min_amp, max_amp = 0.5, 2.0
    return ((size_mm - min_size) / (max_size - min_size)) * (max_amp - min_amp) + min_amp

def size_to_spikewidth(size_mm):
    min_size, max_size = 0.1778, 1.02
    min_width, max_width = 0.0005, 0.002
    return ((size_mm - min_size) / (max_size - min_size)) * (max_width - min_width) + min_width

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

st.markdown(f"**BPFO ≈ {BPFO:.2f} Hz | BPFI ≈ {BPFI:.2f} Hz | BSF ≈ {BSF:.2f} Hz | FTF ≈ {FTF:.2f} Hz**")

# === Fault Signal Generation ===
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
            spike_range = np.arange(center - spike_samples//2, center + spike_samples//2)
            spike_range = spike_range[(spike_range >= 0) & (spike_range < len(t))]
            spike_time = t[spike_range] - t[center]
            impulse = amplitude * np.exp(-((spike_time * fs) ** 2) / 2)
            signal[spike_range] += impulse
    return signal

raw_fault_signal = generate_fault_impulses(f_fault, fs, duration, fault_size_mm)

# === Shaft Modulation and Filtering ===
shaft_signal = 0.3 * np.sin(2 * np.pi * (RPM / 60) * t)
modulated_fault_signal = raw_fault_signal * (1 + shaft_signal)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

filtered_signal = bandpass_filter(modulated_fault_signal, 2000, 5500, fs)

# === Add Noise and Load Modulation ===
load_modulation = 1 + load_variation * np.sin(2 * np.pi * 0.5 * t)
noise = 0.05 * np.random.randn(len(t))
vibration_signal = filtered_signal * load_modulation + noise

# === Envelope Detection and FFT ===
analytic_signal = hilbert(vibration_signal)
envelope = np.abs(analytic_signal)

N = len(t)
freqs = np.fft.rfftfreq(N, 1/fs)
fft_spectrum = np.abs(np.fft.rfft(vibration_signal))
envelope_spectrum = np.abs(np.fft.rfft(envelope))

# === Plotting Function ===
def create_bokeh_plot():
    p1 = figure(title="Time Domain Vibration Signal", x_axis_label="Time [s]", y_axis_label="Amplitude", height=250)
    p1.line(t[:2000], vibration_signal[:2000], line_width=1, legend_label="Signal")
    p1.legend.location = "top_left"

    p2 = figure(title="FFT Spectrum", x_axis_label="Frequency [Hz]", y_axis_label="Amplitude", height=250)
    p2.line(freqs, fft_spectrum, line_width=1, legend_label="FFT Spectrum")
    add_harmonics(p2, f_fault, fs / 2)

    p3 = figure(title="Envelope Spectrum", x_axis_label="Frequency [Hz]", y_axis_label="Amplitude", height=250)
    p3.line(freqs, envelope_spectrum, line_width=1, color="green", legend_label="Envelope Spectrum")
    add_harmonics(p3, f_fault, fs / 2)

    return column(p1, p2, p3)

def add_harmonics(plot, freq, max_freq):
    h = freq
    while h < max_freq:
        vline = Span(location=h, dimension='height', line_color='red', line_dash='dashed', line_width=1)
        plot.add_layout(vline)
        h += freq

# === Display ===
st.bokeh_chart(create_bokeh_plot(), use_container_width=True)
