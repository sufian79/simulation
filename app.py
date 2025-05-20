import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
import pandas as pd

# 5-DOF bearing vibration simulator using solve_ivp
def simulate_5dof_vibration(t, params, fault_size_mm, fault_type, noise_level=0.05):
    """
    Simulate vibration signal from a 5-DOF bearing model with fault size and noise using solve_ivp.

    params: dict with keys - m, c, k, Omega, Fs
    fault_size_mm: float, fault size to scale excitation amplitude
    fault_type: str in ['outer', 'inner', 'ball']
    noise_level: float, noise amplitude

    Returns: vibration signal array (same length as t)
    """
    m = params['m']
    c = params['c']
    k = params['k']
    Omega = params['Omega']  # shaft rotational speed in rad/s
    Fs = params['Fs']

    n = len(t)
    dt = 1 / Fs

    # Fault excitation frequency depending on fault type
    fault_freq_map = {
        'outer': 100,   # example freq in Hz for outer race fault
        'inner': 120,   # example freq for inner race fault
        'ball': 90      # example freq for ball fault
    }
    fault_freq = fault_freq_map.get(fault_type, 100)
    wf = 2 * np.pi * fault_freq  # radians/sec

    # Fault amplitude proportional to fault size
    amp = (fault_size_mm / 1.0) * 1.0  # scale factor

    # Define forcing function at time t
    def F_func(t):
        return amp * np.sin(wf * t)

    # ODE system for one DOF: y = [x, v]
    def bearing_ode(t, y):
        x, v = y
        F = F_func(t)
        dxdt = v
        dvdt = (F - c * v - k * x) / m
        return [dxdt, dvdt]

    # Solve for DOF 0 using solve_ivp
    y0 = [0, 0]  # initial displacement and velocity
    sol = solve_ivp(bearing_ode, (t[0], t[-1]), y0, t_eval=t, method='RK45')

    x0 = sol.y[0]  # displacement for DOF 0

    # For other DOFs, add small coupled vibrations (sine + noise)
    x_others = np.zeros((4, n))
    for dof in range(1, 5):
        freq = fault_freq / (dof + 1)
        x_others[dof - 1, :] = 0.1 * np.sin(2 * np.pi * freq * t) + 0.02 * np.random.randn(n)

    # Sum all DOFs
    signal = x0 + np.sum(x_others, axis=0)

    # Add Gaussian noise
    noise = noise_level * np.random.randn(n)
    signal_noisy = signal + noise

    return signal_noisy

# === Streamlit UI ===
st.title("Bearing Fault Vibration Signal Simulator with 5-DOF Model")

# Sidebar inputs
st.sidebar.header("Bearing Geometry Parameters")
n = st.sidebar.number_input("Number of Rolling Elements (n)", value=7)
R_mm = st.sidebar.number_input("Pitch Diameter (R) [mm]", value=60.0)
d_mm = st.sidebar.number_input("Ball Diameter (d) [mm]", value=20.0)
beta_deg = st.sidebar.number_input("Contact Angle (β) [deg]", value=0.0)
fs = st.sidebar.number_input("Sampling Frequency (fs) [Hz]", value=12000)
duration = st.sidebar.number_input("Signal Duration [s]", value=2.0)
RPM = st.sidebar.number_input("Shaft Speed (RPM)", value=1800)
load_variation = st.sidebar.slider("Load-Induced Modulation", 0.0, 1.0, 0.2)

st.sidebar.header("Fault Parameters")
fault_types = st.sidebar.multiselect("Select Fault Types", ["outer", "inner", "ball"], default=["outer"])
fault_size_mm = st.sidebar.number_input("Fault Size [mm]", value=0.5334, min_value=0.01)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.05)

st.sidebar.header("Simulation Options")
use_5dof = st.sidebar.checkbox("Use 5-DOF Physical Simulation", True)

# Time vector
t = np.linspace(0, duration, int(fs * duration))

# Convert units
R = R_mm / 1000  # meters
d = d_mm / 1000
beta = np.deg2rad(beta_deg)
Omega = RPM * 2 * np.pi / 60  # rad/s

params = {
    'm': 1.0,    # mass in kg (assumed)
    'c': 0.05,   # damping coefficient (assumed)
    'k': 1000.0, # stiffness (assumed)
    'Omega': Omega,
    'Fs': fs
}

# Generate signal either by impulse method or 5-DOF solve_ivp
if use_5dof:
    vibration_signal = np.zeros_like(t)
    for ft in fault_types:
        sig = simulate_5dof_vibration(t, params, fault_size_mm, ft, noise_level=noise_level)
        vibration_signal += sig
else:
    # Impulse-based simplified model
    def size_to_amplitude(size_mm):
        min_size, max_size = 0.1778, 1.02
        min_amp, max_amp = 0.5, 2.0
        return ((size_mm - min_size) / (max_size - min_size)) * (max_amp - min_amp) + min_amp

    def size_to_spikewidth(size_mm):
        min_size, max_size = 0.1778, 1.02
        min_width, max_width = 0.0005, 0.002
        return ((size_mm - min_size) / (max_size - min_size)) * (max_width - min_width) + min_width

    fault_freq_map = {"outer": 100, "inner": 120, "ball": 90}
    def generate_fault_impulses(fault_freq, fs, duration, fault_size_mm):
        amplitude = size_to_amplitude(fault_size_mm)
        spike_width = size_to_spikewidth(fault_size_mm)
        t_local = np.linspace(0, duration, int(fs * duration))
        signal = np.zeros_like(t_local)
        period = 1 / fault_freq
        num_cycles = int(duration / period)
        spike_samples = int(spike_width * fs)
        for i in range(num_cycles):
            center = int(i * period * fs)
            if center + spike_samples < len(t_local):
                spike_range = np.arange(center - spike_samples // 2, center + spike_samples // 2)
                spike_range = spike_range[(spike_range >= 0) & (spike_range < len(t_local))]
                spike_time = t_local[spike_range] - t_local[center]
                impulse = amplitude * np.exp(-((spike_time * fs) ** 2) / 2)
                signal[spike_range] += impulse
        return signal
    
    vibration_signal = np.zeros_like(t)
    for ft in fault_types:
        vibration_signal += generate_fault_impulses(fault_freq_map[ft], fs, duration, fault_size_mm)
    
    vibration_signal += noise_level * np.random.randn(len(t))

# === Envelope detection and FFT ===
analytic_signal = hilbert(vibration_signal)
envelope = np.abs(analytic_signal)

N = len(t)
freqs = np.fft.rfftfreq(N, 1 / fs)
fft_spectrum = np.abs(np.fft.rfft(vibration_signal))
envelope_spectrum = np.abs(np.fft.rfft(envelope))

# === Plotting ===
fig, axs = plt.subplots(3, 1, figsize=(15, 12))

axs[0].plot(t, vibration_signal)
axs[0].set_title(f"Simulated Vibration Signal (Fault Size: {fault_size_mm} mm)")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

axs[1].plot(freqs, fft_spectrum)
axs[1].set_title("Frequency Spectrum")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

axs[2].plot(freqs, envelope_spectrum, color='green')
axs[2].set_title("Envelope Spectrum")
axs[2].set_xlabel("Frequency [Hz]")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True)

fig.tight_layout()
st.pyplot(fig)

# === Download CSV option ===
df = pd.DataFrame({'Time': t, 'Signal': vibration_signal})
csv = df.to_csv(index=False).encode()
st.download_button("Download Signal CSV", csv, "vibration_signal.csv", "text/csv")
