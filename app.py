import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.signal import hilbert, butter, filtfilt

# ----------------------------
# PARAMETERS
# ----------------------------
nb = 9                   # Number of balls
d = 7.94e-3              # Ball diameter (m)
D = 39.32e-3             # Pitch diameter (m)
fs = 10                  # Shaft frequency in Hz
omega_i = 2 * np.pi * fs  # rad/s
omega_c = omega_i * (1 - d / D)
c = 0                    # Clearance
g = 9.81

# Masses (kg)
MP = 12.638
MS = 6.2638
MR = 0.5

# Stiffness and Damping
KP, KS, KR = 15.1e6, 7.42e7, 1e6
RP, RS, RR = 2210.7, 1376.8, 500
Kb = 1.89e10
fault_depth = 1e-3
fault_width = 3e-3
gamma = 1.5

# Faults
FAULT_NONE = 'None'
FAULT_OUTER = 'Outer Race Fault'
FAULT_INNER = 'Inner Race Fault'
FAULT_BALL = 'Ball Fault'

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("Bearing Fault Simulation")
fault_type = st.selectbox("Select Fault Type", [FAULT_NONE, FAULT_OUTER, FAULT_INNER, FAULT_BALL])
noise_level = st.slider("Noise Level (0-1)", 0.0, 1.0, 0.05)
t_end = st.slider("Simulation Time (s)", 0.1, 1.0, 0.5)

# ----------------------------
# FUNCTIONS
# ----------------------------
def ball_angles(t):
    return np.array([(2 * np.pi / nb) * j + omega_c * t for j in range(nb)])

def apply_fault(delta, theta_j, t):
    if fault_type == FAULT_OUTER:
        fault_angle = 0.0
        if fault_angle < theta_j < fault_angle + fault_width:
            delta -= fault_depth
    elif fault_type == FAULT_INNER:
        fault_angle = omega_i * t
        if fault_angle < theta_j < fault_angle + fault_width:
            delta -= fault_depth
    elif fault_type == FAULT_BALL:
        beta = theta_j % (2 * np.pi)
        if 0 < beta < fault_width:
            delta -= fault_depth
    return delta

def contact_force(xd, yd, t):
    theta = ball_angles(t)
    Fx, Fy = 0.0, 0.0
    for j in range(nb):
        tj = theta[j]
        delta_j = xd * np.cos(tj) + yd * np.sin(tj) - c
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
    xodotdot = (-RP * xodot - KP * xo + Fx) / MP
    yodotdot = (-RP * yodot - KP * yo - KR * (yo - yr) - RR * (yodot - yrdot) + Fy - MP * g) / MP
    xidotdot = (-RS * xidot - KS * xi - Fx) / MS
    yidotdot = (-RS * yidot - KS * yi - Fy - MS * g) / MS
    yrdotdot = (-KR * (yr - yo) - RR * (yrdot - yodot) - MR * g) / MR
    return [xodot, yodot, xidot, yidot, yrdot, xodotdot, yodotdot, xidotdot, yidotdot, yrdotdot]

# ----------------------------
# SIMULATE
# ----------------------------
y0 = [0]*10
fsamp = 50000
t_eval = np.linspace(0, t_end, int(fsamp * t_end))
sol = solve_ivp(bearing_5dof, [0, t_end], y0, t_eval=t_eval, method='RK45')
time = sol.t
yo = sol.y[1] + noise_level * np.random.normal(0, 1e-6, len(sol.y[1]))

# Envelope analysis
analytic_signal = hilbert(yo)
envelope = np.abs(analytic_signal)

# FFT
def bandpass_filter(signal, fs, lowcut=1000, highcut=10000):
    b, a = butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

filtered = bandpass_filter(yo, fsamp)
fft_vals = np.abs(np.fft.rfft(filtered))
freqs = np.fft.rfftfreq(len(filtered), 1/fsamp)

# ----------------------------
# PLOT
# ----------------------------
st.subheader("Raw Vibration Signal (Outer Race Displacement)")
st.line_chart(y=yo, x=time)

st.subheader("Envelope of Vibration Signal")
st.line_chart(y=envelope, x=time)

st.subheader("FFT of Filtered Signal (Frequency Domain)")
fig, ax = plt.subplots()
ax.plot(freqs, fft_vals)
ax.set_xlim(0, 2000)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Amplitude')
ax.set_title('FFT Spectrum')
st.pyplot(fig)

st.success("Simulation complete with fault: {}".format(fault_type))
