import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from scipy.signal import find_peaks, savgol_filter

import sys

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

G = 6.67430e-11

# Contrived Sun–Earth–Mars system
central_mass = 300000.0
r1, r2 = 1.0, 2
m1 = 1.0
m2 = 0.107
v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2)

# === base example
sim.add(m=central_mass)   # Sun
sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
sim.add(m=m2, x=r2, y=0, vy=v2)  # Mars
t_max = 2e4

# === elliptical inside
# sim.add(m=central_mass)   # Sun
# sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
# sim.add(m=1, a=2, e=0.7) 
# t_max = 3e4

# === Time Setup ===
# t_max = 2e4
n_steps = int(1e4)
times = np.linspace(0, t_max, n_steps)
dt = times[1] - times[0]

# === Integrate and Record Positions ===
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

# === Compute Accelerations and Velocities ===
accel_earth_approx = np.zeros_like(positions_1)
for i in range(2, n_steps):
    accel_earth_approx[i - 1] = (positions_1[i] - 2 * positions_1[i - 1] + positions_1[i - 2]) / dt**2

accel_sun = np.zeros_like(positions_1)
for i in range(n_steps):
    r_vec = positions_1[i] - positions_0[i]
    r = np.linalg.norm(r_vec)
    if r > 0:
        accel_sun[i] = -G * central_mass * r_vec / r**3

# Perturbation acceleration
accel_mars = accel_earth_approx - accel_sun

# === Compute Correlation and Ratio ===
corr_vals = np.zeros(n_steps)
ratio_vals = np.zeros(n_steps)
for i in range(n_steps):
    a = accel_mars[i]
    b = positions_2[i] - positions_1[i]
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        corr_vals[i] = np.nan
    else:
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        corr_vals[i] = np.dot(a_norm, b_norm)

    mag_accel_sun = np.linalg.norm(accel_sun[i])
    mag_accel_net = np.linalg.norm(accel_earth_approx[i])
    ratio_vals[i] = mag_accel_sun / mag_accel_net if mag_accel_net > 0 else np.nan

smoothing = 5
smoothed_ratio_vals = np.convolve(ratio_vals, (np.zeros(smoothing)+1)/smoothing, "same")

# === Compute Ratio 1st and 2nd Derivative ===
dt_ratio = np.zeros(len(ratio_vals))
dt2_ratio = np.zeros(len(ratio_vals))

for i in range(len(ratio_vals)): 
    if (i >= 1):
        dt_ratio[i-1] = (ratio_vals[i] - ratio_vals[i-1]) / dt
    if (i >= 2):
        dt2_ratio[i-2] = (ratio_vals[i] - 2*ratio_vals[i-1] + ratio_vals[i-2]) / dt**2
    if dt2_ratio[i] == 0:
        dt2_ratio[i] = np.nan
    if dt_ratio[i] == 0:
        dt_ratio[i] = np.nan

# === Peak Detection Using Derivative Information ===

# Smooth over the data
smoothing = 5
dt_ratio = np.convolve(dt_ratio, (np.zeros(smoothing)+1)/smoothing, "same")
dt2_ratio = np.convolve(dt2_ratio, (np.zeros(smoothing)+1)/smoothing, "same")

# How far ahead to approximate dt_ratio
euler_step_size = 10

# Threshold for dt_height check
dt_threshold = 0.05    # % range of dt_ratio

# Threshold for dt2_height check

crossings = np.zeros(len(ratio_vals))
for i in range(len(dt_ratio)):
    if dt_ratio[i] > dt_threshold*(np.nanmax(dt_ratio)-np.nanmin(dt_ratio)) and dt_ratio[i] + euler_step_size*dt2_ratio[i] < 0:
        crossings[i] = times[i]

# === Plot Setup ===
fig, (ax_orbit, ax_combined, ax_delta) = plt.subplots(3, 1, figsize=(6, 9))
plt.subplots_adjust(bottom=0.25, hspace=0.35)
idx = 0 # Animation starts on this frame

ax_delta.set_xlim(times[0], times[-1])
ax_delta.plot(times, np.zeros(len(times)), "--", color="black")
ax_delta.plot(times, dt_ratio)
ax_delta.plot(times, dt2_ratio, color = "purple")
ax_delta.scatter(crossings, np.zeros_like(crossings), color = "red")


# 1️⃣ Orbit Plot
# ax_orbit.set_xlim(-1.6, 1.6)
# ax_orbit.set_ylim(-1.6, 1.6)
ax_orbit.set_aspect('equal')
ax_orbit.set_title("Orbital Motion with Normalized Perturbation and Earth–Mars Vectors")
ax_orbit.grid(True)

ax_orbit.plot(positions_1[:, 0], positions_1[:, 1], 'b-', alpha=0.3)
ax_orbit.plot(positions_2[:, 0], positions_2[:, 1], 'r-', alpha=0.3)
ax_orbit.plot(positions_0[:, 0], positions_0[:, 1], color='gold', lw=0.5, alpha=0.3)


marker_sun, = ax_orbit.plot([], [], 'yo', markersize=8)
marker_earth, = ax_orbit.plot([], [], 'bo', markersize=5)
marker_mars, = ax_orbit.plot([], [], 'ro', markersize=5)
pert_arrow = None
mars_arrow = None
ax_orbit.legend(loc="upper right")

# 2️⃣ Combined Cosine & Ratio Plot
ax_combined.set_xlim(times[0], times[-1])
ax_combined.set_title("Cosine (Magenta) and Accel Ratio (Green)")
ax_combined.set_xlabel("Time (s)")

# Left y-axis for cosine
ax_combined.set_ylabel("Cosine", color='m')
ax_combined.set_ylim(-1.1, 1.1)
corr_line, = ax_combined.plot([], [], 'm-', label="Cosine (pert vs Mars)")

# Right y-axis for ratio
scale = 1e3
ratio_vals = (ratio_vals-1) * scale
ax_ratio_twin = ax_combined.twinx()
ax_ratio_twin.set_ylabel("Accel Ratio |Sun|/|Net|", color='g')
ax_ratio_twin.set_ylim(0.9*np.nanmin(ratio_vals), 1.1*np.nanmax(ratio_vals))
ratio_line, = ax_ratio_twin.plot([], [], 'g-', label="Accel Ratio")

# Find points of interest in ratio plot
ratio_peaks, _ = find_peaks(ratio_vals)
peak_time = times[ratio_peaks]
peak_height = ratio_vals[ratio_peaks]
ax_ratio_twin.scatter(peak_time, peak_height)

# Vertical time marker
corr_time_marker = ax_combined.axvline(times[0], color='k', ls='--')


# === Update Function ===
def update(i):
    global pert_arrow, mars_arrow
    marker_sun.set_data([positions_0[i, 0]], [positions_0[i, 1]])
    marker_earth.set_data([positions_1[i, 0]], [positions_1[i, 1]])
    marker_mars.set_data([positions_2[i, 0]], [positions_2[i, 1]])

    for arrow in [pert_arrow, mars_arrow]:
        if arrow is not None:
            arrow.remove()
    pert_arrow = mars_arrow = None

    # Normalized arrows from Earth
    earth_pos = positions_1[i]
    pert_vec = accel_mars[i]
    mars_vec = positions_2[i] - positions_1[i]
    if np.linalg.norm(pert_vec) > 0 and np.linalg.norm(mars_vec) > 0:
        pert_unit = pert_vec / np.linalg.norm(pert_vec)
        mars_unit = mars_vec / np.linalg.norm(mars_vec)
        arrow_len = 0.3
        pert_arrow = ax_orbit.arrow(earth_pos[0], earth_pos[1],
                                    pert_unit[0]*arrow_len, pert_unit[1]*arrow_len,
                                    color='lime', head_width=0.03)
        mars_arrow = ax_orbit.arrow(earth_pos[0], earth_pos[1],
                                    mars_unit[0]*arrow_len, mars_unit[1]*arrow_len,
                                    color='red', head_width=0.03)

    corr_line.set_data(times[:i], corr_vals[:i])
    ratio_line.set_data(times[:i], (ratio_vals[:i]-1)+1)
    corr_time_marker.set_xdata([times[i], times[i]])

# === Slider and Button ===
ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
slider = Slider(ax_slider, 'Time', 0, n_steps - 1, valinit=n_steps-1, valstep=1)

paused = False
def toggle(event):
    global paused
    paused = not paused

ax_button = plt.axes([0.86, 0.11, 0.1, 0.04])
button = Button(ax_button, 'Play/Pause')
button.on_clicked(toggle)

def slider_update(val):
    global idx
    idx = int(slider.val)
    update(idx)
slider.on_changed(slider_update)

# === Textbox for Jump-to-Frame ===
axbox = plt.axes([0.1, 0.01, 0.2, 0.075])
text_box = TextBox(axbox, 'Jump to Frame:', initial="0")

def jump_to_frame(text):
    try:
        global idx
        idx = int(text)
        update(idx)         # update the plots to the current time value
        slider.set_val(idx) # update the slider to be on the current time value
    except:
        print("Invalid frame index")

text_box.on_submit(jump_to_frame)


def animate(frame):
    global idx
    if not paused:
        idx = (idx + 1) % n_steps   # Increment counter and wrap around at the end 
        slider.set_val(idx)
    return []

ani = FuncAnimation(fig, animate, frames=times, interval=10, blit=True, repeat=True)

plt.show()
