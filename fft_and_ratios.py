import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from scipy.signal import find_peaks

# === Utility ===
def moving_average(a, n=3):
    weights = np.ones(n) / n
    return np.convolve(a, weights, mode='valid')

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"
G = 6.67430e-11

# Simple Sun–Earth–Mars-like system
central_mass = 300000.0
sim.add(m=central_mass)
r1, r2 = 1.0, 2.0
m1, m2 = 1.0, 0.107
v1, v2 = np.sqrt(G * central_mass / r1), np.sqrt(G * central_mass / r2)
sim.add(m=m1, x=r1, y=0, vy=v1)
sim.add(m=m2, x=r2, y=0, vy=v2)

# === Time Setup ===
t_max = 2e4
n_steps = int(2e5)
times = np.linspace(0, t_max, n_steps)

# === Integrate Orbits ===
positions_0 = np.zeros((n_steps, 2))
positions_1 = np.zeros((n_steps, 2))
positions_2 = np.zeros((n_steps, 2))
sim_copy = sim.copy()
for i, t in enumerate(times):
    sim_copy.integrate(t)
    p0, p1, p2 = sim_copy.particles
    positions_0[i] = [p0.x, p0.y]
    positions_1[i] = [p1.x, p1.y]
    positions_2[i] = [p2.x, p2.y]

# === Compute Accelerations ===
dt = times[1] - times[0]
accel_earth_approx = np.zeros_like(positions_1)
for i in range(2, n_steps):
    accel_earth_approx[i - 1] = (positions_1[i] - 2 * positions_1[i - 1] + positions_1[i - 2]) / dt**2

accel_sun = np.zeros_like(positions_1)
for i in range(n_steps):
    r_vec = positions_1[i] - positions_0[i]
    r = np.linalg.norm(r_vec)
    if r > 0:
        accel_sun[i] = -G * central_mass * r_vec / r**3

accel_mars = accel_earth_approx - accel_sun

# === Initial parameters ===
mov_avg_len = 3
prominence_percent = 0.05

def compute_metrics(mov_avg_len, prominence_percent):
    # Moving average smoothing
    ratio_vals = np.zeros(n_steps)
    for i in range(n_steps):
        mag_accel_sun = np.linalg.norm(accel_sun[i])
        mag_accel_net = np.linalg.norm(accel_earth_approx[i])
        ratio_vals[i] = mag_accel_sun / mag_accel_net if mag_accel_net > 0 else np.nan
    ratio_vals = (ratio_vals - 1) * central_mass + 1
    ratio_vals = moving_average(ratio_vals, int(mov_avg_len))
    times_ratio = times[:len(ratio_vals)]

    # Correlation (cosine similarity)
    corr_vals = np.zeros(len(times_ratio))
    for i in range(len(times_ratio)):
        a = accel_mars[i]
        b = positions_2[i] - positions_1[i]
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            corr_vals[i] = np.nan
        else:
            corr_vals[i] = np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))

    # Peak detection in time domain
    valid = ~np.isnan(ratio_vals)
    prominence_value = prominence_percent * (np.nanmax(ratio_vals) - np.nanmin(ratio_vals))
    peaks, _ = find_peaks(ratio_vals[valid], prominence=prominence_value, distance=2000)
    valid_times = times_ratio[valid]
    peak_times = valid_times[peaks]
    peak_values = ratio_vals[valid][peaks]
    return times_ratio, corr_vals, ratio_vals, peak_times, peak_values

# Compute initial metrics
times_ratio, corr_vals, ratio_vals, peak_times, peak_values = compute_metrics(mov_avg_len, prominence_percent)

# === Plot Setup ===
fig, (ax_ratio, ax_fft) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(bottom=0.25, hspace=0.35)  # Leave space at bottom for TextBox

# Time domain plot setup
ax_ratio.set_title("Cosine and Accel Ratio over Time")
ax_ratio.set_xlabel("Time (s)")
ax_ratio.set_ylabel("Value")
corr_line, = ax_ratio.plot([], [], 'm-', label='Cosine')
ratio_line, = ax_ratio.plot([], [], 'g-', label='Accel Ratio')
peak_dots, = ax_ratio.plot([], [], 'ro', markersize=5, label='Peaks')
ax_ratio.legend()
ax_ratio.grid(True)

# Frequency domain plot setup
ax_fft.set_title("Magnitude Spectrum of Smoothed Accel Ratio")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Magnitude")
fft_line, = ax_fft.plot([], [], 'b-', label='FFT Magnitude')
fft_peak_dots, = ax_fft.plot([], [], 'ro', markersize=5, label='FFT Peaks')
ax_fft.legend()
ax_fft.grid(True)

# === Update function ===
def update_all_plots():
    global times_ratio, corr_vals, ratio_vals, peak_times, peak_values

    # Update time domain plots
    corr_line.set_data(times_ratio, corr_vals)
    ratio_line.set_data(times_ratio, ratio_vals)
    peak_dots.set_data(peak_times, peak_values)
    ax_ratio.set_xlim(times_ratio[0], times_ratio[-1])
    ax_ratio.set_ylim(min(np.nanmin(corr_vals), np.nanmin(ratio_vals)) * 0.95,
                      max(np.nanmax(corr_vals), np.nanmax(ratio_vals)) * 1.05)

    # Compute FFT on cleaned ratio_vals
    cleaned_ratio = np.nan_to_num(ratio_vals, nan=0.0, posinf=0.0, neginf=0.0)
    fft_vals = np.fft.rfft(cleaned_ratio - np.mean(cleaned_ratio))
    fft_freq = np.fft.rfftfreq(len(cleaned_ratio), d=(times_ratio[1] - times_ratio[0]))
    fft_magnitude = np.abs(fft_vals)

    fft_line.set_data(fft_freq, fft_magnitude)

    # Detect peaks in frequency domain
    if len(fft_magnitude) > 0:
        fft_prominence = 0.05 * (np.max(fft_magnitude) - np.min(fft_magnitude))
        fft_peaks, _ = find_peaks(fft_magnitude, prominence=fft_prominence)
        fft_peak_freqs = fft_freq[fft_peaks]
        fft_peak_vals = fft_magnitude[fft_peaks]
        fft_peak_dots.set_data(fft_peak_freqs, fft_peak_vals)

        if len(fft_peak_freqs) > 0:
            max_peak_freq = np.max(fft_peak_freqs)
            xlim_max = min(5 * max_peak_freq, np.max(fft_freq))
        else:
            xlim_max = np.max(fft_freq)
    else:
        fft_peak_dots.set_data([], [])
        xlim_max = np.max(fft_freq)

    ax_fft.set_xlim(0, xlim_max)
    ax_fft.set_ylim(0, 1.1 * np.max(fft_magnitude))

    fig.canvas.draw_idle()


# Initial plot update
update_all_plots()

# === Moving Average Length TextBox ===
ax_ma_len = plt.axes([0.3, 0.1, 0.1, 0.05])  # x, y, width, height in figure coords
text_box = TextBox(ax_ma_len, 'MA len (odd int)', initial=str(mov_avg_len))

def on_ma_len_submit(text):
    global mov_avg_len, times_ratio, corr_vals, ratio_vals, peak_times, peak_values

    try:
        val = int(text)
        if val < 1:
            val = 1
        if val % 2 == 0:
            val += 1  # force odd
        mov_avg_len = val
    except ValueError:
        # ignore invalid input, reset text box to current value
        text_box.set_val(str(mov_avg_len))
        return

    # Recompute metrics and update plots
    times_ratio, corr_vals, ratio_vals, peak_times, peak_values = compute_metrics(mov_avg_len, prominence_percent)
    update_all_plots()

text_box.on_submit(on_ma_len_submit)

plt.show()
