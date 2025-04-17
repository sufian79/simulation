import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import hilbert, butter, filtfilt

# ----------------------------
# STREAMLIT SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("Simulation Settings")

# Select bearing model
bearing_model = st.sidebar.selectbox("Select Bearing Model", ['Custom', '6000', '6200', '6300'])

# Masses
masses = {
    "pedestal": st.sidebar.number_input("Mass (Pedestal) [kg]", 0.1, 50.0, 12.638),
    "shaft": st.sidebar.number_input("Mass (Shaft) [kg]", 0.1, 50.0, 6.2638),
    "sprung": st.sidebar.number_input("Mass (Sprung) [kg]", 0.1, 10.0, 0.5)
}

# Stiffness
stiffness = {
    "pedestal": st.sidebar.number_input("Stiffness (Pedestal) [N/m]", 1e3, 1e8, 15.1e6),
    "shaft": st.sidebar.number_input("Stiffness (Shaft) [N/m]", 1e3, 1e8, 7.42e7),
    "sprung": st.sidebar.number_input("Stiffness (Sprung) [N/m]", 1e3, 1e8, 1e6)
}

# Damping
damping = {
    "pedestal": st.sidebar.number_input("Damping (Pedestal) [Ns/m]", min_value=1.0, max_value=1e4, value=2210.7),
    "shaft": st.sidebar.number_input("Damping (Shaft) [Ns/m]", min_value=1.0, max_value=1e4, value=1376.8),
    "sprung": st.sidebar.number_input("Damping (Sprung) [Ns/m]", min_value=1.0, max_value=1e4, value=500.0)
}

# Ball contact
Kb = st.sidebar.number_input("Ball Stiffness Kb [N/m^(3/2)]", 1e6, 1e11, 1.89e10)
gamma = 1.5

# Bearing geometry options for common bearings
bearing_geometry_dict = {
    "6000": {"num_balls": 9, "ball_diameter": 7.94e-3, "bearing_diameter": 39.32e-3, "shaft_speed": 10},
    "6200": {"num_balls": 10, "ball_diameter": 8.0e-3, "bearing_diameter": 42.0e-3, "shaft_speed": 12},
    "6300": {"num_balls": 12, "ball_diameter": 8.5e-3, "bearing_diameter": 45.0e-3, "shaft_speed": 15}
}

if bearing_model == 'Custom':
    bearing_geometry = {
        "num_balls": st.sidebar.number_input("Number of Balls", 1, 50, 9),
        "ball_diameter": st.sidebar.number_input("Ball Diameter [m]", 1e-3, 1e-1, 7.94e-3),
        "bearing_diameter": st.sidebar.number_input("Bearing Diameter [m]", 1e-3, 1e-1, 39.32e-3),
        "shaft_speed": st.sidebar.number_input("Shaft Speed [Hz]", 1, 100, 10)
    }
else:
    bearing_geometry = bearing_geometry_dict[bearing_model]

# Additional parameters derived from the geometry
bearing_geometry["omega_i"] = 2 * np.pi * bearing_geometry["shaft_speed"]
bearing_geometry["omega_c"] = bearing_geometry["omega_i"] * (1 - bearing_geometry["ball_diameter"] / bearing_geometry["bearing_diameter"])
bearing_geometry["clearance"] = 0
bearing_geometry["g"] = 9.81  # gravity in m/s^2

# ----------------------------
# FUNCTIONS
# ----------------------------
def ball_angles(t):
    return np.array([(2 * np.pi / bearing_geometry["num_balls"]) * j + bearing_geometry["omega_c"] * t for j in range(bearing_geometry["num_balls"])])

def apply_fault(delta, theta_j, t):
    if fault_type == 'Outer Race Fault':
        fault_angle = 0.0
        if fault_angle < theta_j < fault_angle + fault_width:
            delta -= fault_depth
    elif fault_type == 'Inner Race Fault':
        fault_angle = bearing_geometry["omega_i"] * t
        if fault_angle < theta_j < fault_angle + fault_width:
            delta -= fault_depth
    elif fault_type == 'Ball Fault':
        beta = theta_j % (2 * np.pi)
        if 0 < beta < fault_width:
            delta -= fault_depth
    return delta

def contact_force(xd, yd, t):
    theta = ball_angles(t)
    Fx, Fy = 0.0, 0.0
    for j in range(bearing_geometry["num_balls"]):
        tj = theta[j]
        delta_j = xd * np.cos(tj) + yd * np.sin(tj) - bearing_geometry["clearance"]
        delta_j = apply_fault(delta_j, tj, t)
        if delta_j > 0:
            F = Kb * delta_j**gamma
            Fx += F * np.cos(tj)
            Fy += F * np.sin(tj)
    return Fx, Fy

def bearing_5dof(t, y):
    xo, yo, xi, yi, yr, xodot, yodot, xidot, yidot, yrdot = y
    xd = xi - xo
    yd = yi - yo
    Fx, Fy = contact_force(xd, yd, t)
    xodotdot = (-damping["pedestal"] * xodot - stiffness["pedestal"] * xo + Fx) / masses["pedestal"]
    yodotdot = (-damping["pedestal"] * yodot - stiffness["pedestal"] * yo - stiffness["sprung"] * (yo - yr) - damping["sprung"] * (yodot - yrdot) + Fy - masses["pedestal"] * bearing_geometry["g"]) / masses["pedestal"]
    xidotdot = (-damping["shaft"] * xidot - stiffness["shaft"] * xi - Fx) / masses["shaft"]
    yidotdot = (-damping["shaft"] * yidot - stiffness["shaft"] * yi - Fy - masses["shaft"] * bearing_geometry["g"]) / masses["shaft"]
    yrdotdot = (-stiffness["sprung"] * (yr - yo) - damping["sprung"] * (yrdot - yodot) - masses["sprung"] * bearing_geometry["g"]) / masses["sprung"]
    return [xodot, yodot, xidot, yidot, yrdot, xodotdot, yodotdot, xidotdot, yidotdot, yrdotdot]

def bandpass_filter(signal, fs, lowcut=1000, highcut=10000):
    b, a = butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

# ----------------------------
# SIMULATE
# ----------------------------
y0 = [0]*10
t_eval = np.linspace(0, t_end, int(fsamp * t_end))
sol = solve_ivp(bearing_5dof, [0, t_end], y0, t_eval=t_eval, method='RK45')
time = sol.t
yo = sol.y[1] + noise_level * np.random.normal(0, 1e-6, len(sol.y[1]))

# Envelope and FFT
analytic_signal = hilbert(yo)
envelope = np.abs(analytic_signal)
filtered = bandpass_filter(yo, fsamp)
fft_vals = np.abs(np.fft.rfft(filtered))
freqs = np.fft.rfftfreq(len(filtered), 1/fsamp)

# ----------------------------
# PLOT
# ----------------------------
st.title("Bearing Fault Simulation")

# Show signal only after stabilization
stable_idx = time >= 0.1

st.subheader("Raw Vibration Signal")
df_raw = pd.DataFrame({'Time': time[stable_idx], 'Signal': yo[stable_idx]})
st.line_chart(df_raw.set_index('Time'))

st.subheader("Envelope Signal")
df_env = pd.DataFrame({'Time': time[stable_idx], 'Envelope': envelope[stable_idx]})
st.line_chart(df_env.set_index('Time'))

st.subheader("FFT Spectrum")
fig, ax = plt.subplots()
ax.plot(freqs, fft_vals)
ax.set_xlim(0, 2000)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Amplitude')
ax.set_title('FFT of Filtered Signal')
st.pyplot(fig)

st.success(f"Simulation complete with fault: {fault_type}")
