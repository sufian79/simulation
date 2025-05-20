import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import butter, lfilter, hilbert
import pandas as pd
from io import BytesIO

# Multi-DOF bearing vibration simulator using solve_ivp
def simulate_mdof_vibration(t, params, fault_size_mm, fault_type, noise_level=0.05):
    """
    Realistic 5-DOF MDOF bearing vibration simulator using solve_ivp.

    Parameters:
        t : np.ndarray
            Time vector
        params : dict
            Dictionary with 'm', 'c', 'k', 'Omega', 'Fs'
        fault_size_mm : float
            Fault size (mm)
        fault_type : str
            One of ['outer', 'inner', 'ball']
        noise_level : float
            Noise amplitude

    Returns:
        signal_noisy : np.ndarray
            Noisy vibration signal (summed across DOFs)
    """
    n_dof = 5
    m = params.get('m', 1.0)       # base mass per DOF
    c = params.get('c', 0.05)      # base damping per DOF
    k = params.get('k', 1000.0)    # base stiffness per DOF
    Fs = params['Fs']

    # Assign physical meaning to each DOF
    # DOF 0: Shaft
    # DOF 1: Inner Race
    # DOF 2: Rolling Element
    # DOF 3: Outer Race
    # DOF 4: Bearing Housing

    # Mass matrix (in kg)
    M = np.diag([m, 0.8*m, 0.3*m, 0.9*m, 1.2*m])

    # Stiffness matrix (N/m)
    K = np.array([
        [2*k,   -k,     0,     0,     0],
        [-k,  2.5*k,   -k,     0,     0],
        [0,    -k,    2*k,   -k,     0],
        [0,     0,    -k,   2.2*k,  -k],
        [0,     0,     0,   -k,   2*k]
    ])

    # Damping matrix (Ns/m)
    C = np.array([
        [2*c,   -c,     0,     0,     0],
        [-c,  2.5*c,   -c,     0,     0],
        [0,    -c,    2*c,   -c,     0],
        [0,     0,    -c,   2.2*c,  -c],
        [0,     0,     0,   -c,   2*c]
    ])

    # Fault excitation frequency
    fault_freq_map = {"outer": BPFO, "inner": BPFI, "ball": BSF}
    f_fault = fault_freq_map.get(fault_type, BPFO)
    wf = 2 * np.pi * f_fault
    amp = fault_size_mm * 10  # Fault size scaled to force

    # Define external force vector
    def F_func(t_local):
        F = np.zeros(n_dof)
        if fault_type == "outer":
            F[3] = amp * np.sin(wf * t_local)  # outer race
        elif fault_type == "inner":
            F[1] = amp * np.sin(wf * t_local)  # inner race
        elif fault_type == "ball":
            F[2] = amp * np.sin(wf * t_local)  # ball element
        return F

    # ODE definition
    def ode(t_local, y):
        x = y[:n_dof]
        v = y[n_dof:]
        a = np.linalg.solve(M, F_func(t_local) - C @ v - K @ x)
        return np.concatenate((v, a))

    y0 = np.zeros(2 * n_dof)
    sol = solve_ivp(ode, (t[0], t[-1]), y0, t_eval=t, method='RK45')

    # DOF responses
    x_all = sol.y[:n_dof, :]
    signal = np.sum(x_all, axis=0)

    # Add Gaussian noise
    signal_noisy = signal + noise_level * np.random.randn(len(signal))
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

# === Fault Selection ===
st.sidebar.header("Fault Parameters")
fault_types = st.sidebar.multiselect("Select Fault Types", ["outer", "inner", "ball"], default=["outer"])
fault_size_mm = st.sidebar.number_input("Fault Size [mm]", value=0.5334)
fault_depth = st.sidebar.slider("Fault Depth (for future use)", 0.0, 1.0, 0.5)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.05)

# === Visualization Options ===
show_envelope = st.sidebar.checkbox("Show Envelope Spectrum", True)
show_harmonics = st.sidebar.checkbox("Mark Harmonics on Spectrum", True)

st.sidebar.header("Simulation Options")
use_5dof = st.sidebar.checkbox("Use 5-DOF Physical Simulation", True)

st.sidebar.header("System Parameters (5-DOF)")
m = st.sidebar.number_input("Mass per DOF (m) [kg]", value=1.0)
c = st.sidebar.number_input("Damping Coefficient (c) [Ns/m]", value=0.05)
k = st.sidebar.number_input("Stiffness (k) [N/m]", value=1000.0)

# Time vector
t = np.linspace(0, duration, int(fs * duration))

# Convert units
R = R_mm / 1000  # meters
d = d_mm / 1000
beta = np.deg2rad(beta_deg)
Omega = RPM * 2 * np.pi / 60  # rad/s

params = {
    'm': m,    # mass in kg (assumed)
    'c': c,   # damping coefficient Ns/m (assumed)
    'k': k, # stiffness N/m (assumed)
    'Omega': Omega,
    'Fs': fs     # Hz
}

# === Bearing Frequencies ===
def bearing_fault_frequencies(n, d, R, beta, RPM):
    fr = RPM / 60
    FTF = 0.5 * fr * (1 - (d / R) * np.cos(beta))
    BPFI = 0.5 * n * fr * (1 + (d / R) * np.cos(beta))
    BPFO = 0.5 * n * fr * (1 - (d / R) * np.cos(beta))
    BSF = R / d * fr * (1 - ((d / R * np.cos(beta)) ** 2))
    return FTF, BPFI, BPFO, BSF
    
FTF, BPFI, BPFO, BSF = bearing_fault_frequencies(n, d, R, beta, RPM)
st.write(f"**BPFO**: {BPFO:.2f} Hz | **BPFI**: {BPFI:.2f} Hz | **BSF**: {BSF:.2f} Hz | **FTF**: {FTF:.2f} Hz")
fault_freq_map = {"outer": BPFO, "inner": BPFI, "ball": BSF}

# Generate signal either by impulse method or 5-DOF solve_ivp
if use_5dof:
    vibration_signal = np.zeros_like(t)
    for ft in fault_types:
        sig = simulate_mdof_vibration(t, params, fault_size_mm, ft, noise_level=noise_level)
        vibration_signal += sig
else:
    # === Fault Size Mapping ===
    def size_to_amplitude(size_mm):
        min_size, max_size = 0.1778, 1.02
        min_amp, max_amp = 0.5, 2.0
        return ((size_mm - min_size) / (max_size - min_size)) * (max_amp - min_amp) + min_amp
    
    def size_to_spikewidth(size_mm):
        min_size, max_size = 0.1778, 1.02
        min_width, max_width = 0.0005, 0.002
        return ((size_mm - min_size) / (max_size - min_size)) * (max_width - min_width) + min_width
        
    # === Generate Impulses ===
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
    
    # === Composite Fault Signal ===
    raw_fault_signal = np.zeros_like(t)
    
    for ft in fault_types:
        raw_fault_signal += generate_fault_impulses(fault_freq_map[ft], fs, duration, fault_size_mm)
    
    shaft_signal = 0.3 * np.sin(2 * np.pi * (RPM / 60) * t)
    modulated_fault_signal = raw_fault_signal * (1 + shaft_signal)
    
    # === Bandpass Filter ===
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = fs / 2
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return lfilter(b, a, data)
    
    filtered_signal = bandpass_filter(modulated_fault_signal, 2000, 5500, fs)
    
    # === Add Noise and Modulation ===
    load_modulation = 1 + load_variation * np.sin(2 * np.pi * 0.5 * t)
    noise = 0.05 * np.random.randn(len(t))
    vibration_signal = filtered_signal * load_modulation + noise

# === Envelope detection and FFT ===
analytic_signal = hilbert(vibration_signal)
envelope = np.abs(analytic_signal)

N = len(t)
freqs = np.fft.rfftfreq(N, 1 / fs)
fft_spectrum = np.abs(np.fft.rfft(vibration_signal))
envelope_spectrum = np.abs(np.fft.rfft(envelope))

# === Plotting ===
fig, axs = plt.subplots(3 if show_envelope else 2, 1, figsize=(15, 10))

axs[0].plot(t, vibration_signal, label="Time-domain Signal")
axs[0].set_title(f"Composite Fault Signal - Fault Size {fault_size_mm} mm")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True)

def plot_fault_harmonics(ax, freqs_to_mark, max_freq):
    for freq in freqs_to_mark:
        h = freq
        while h < max_freq:
            ax.axvline(h, color='red', linestyle='--', linewidth=1)
            h += freq

axs[1].plot(freqs, fft_spectrum, label="FFT Spectrum")
if show_harmonics:
    selected_freqs = [fault_freq_map[ft] for ft in fault_types]
    plot_fault_harmonics(axs[1], selected_freqs, fs / 2)
axs[1].set_title("Frequency Spectrum")
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True)

if show_envelope:
    axs[2].plot(freqs, envelope_spectrum, color='green', label="Envelope Spectrum")
    if show_harmonics:
        plot_fault_harmonics(axs[2], selected_freqs, fs / 2)
    axs[2].set_title("Envelope Spectrum")
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()
    axs[2].grid(True)

fig.tight_layout()
st.pyplot(fig)

# === Download CSV Option ===
df = pd.DataFrame({'Time': t, 'Signal': vibration_signal})
csv = df.to_csv(index=False).encode()
st.download_button("Download Signal CSV", csv, "vibration_signal.csv", "text/csv")

# === Placeholder ML Prediction ===
if st.button("Predict Fault Type (Mock)"):
    if fault_types:
        st.success(f"Predicted: {', '.join([ft.capitalize() for ft in fault_types])} Fault(s)")
    else:
        st.warning("No fault type selected for prediction.")
