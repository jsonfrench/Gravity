import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from scipy.signal import find_peaks
from scipy.stats import zscore
from matplotlib.patches import Wedge

import sys

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

# G = 6.67430e-11
G = sim.G

# Contrived Sun–Earth–Mars system
central_mass = 300000.0
r1, r2 = 1.0, 2
m1 = 1.0
m2 = 0.107 * 1000
v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2) 

# # === base example
sim.add(m=central_mass)   # Sun
sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
sim.add(m=m2, x=r2, y=0, vy=v2)  # Mars
t_max = 2e5
num_steps = 1e4
num_bins = 50

# === elliptical inside
# sim.add(m=central_mass)   # Sun
# sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
# sim.add(m=1, a=2, e=0.7) 
# t_max = 3e4

# # === Real world example
# rebound.horizons.SSL_CONTEXT = "unverified"
# sim.add("Sun")   # Sun
# sim.add("Earth")  # Earth
# sim.add("Mars")  # Mars
# t_max = 60*60*24*365.25*20


# === Time Setup ===
# t_max = 2e4
n_steps = int(num_steps)
times = np.linspace(0, t_max, n_steps)
dt = times[1] - times[0]

# === Integrate and Record Positions ===
positions_0 = np.zeros((n_steps, 2))
positions_1 = np.zeros((n_steps, 2))
positions_2 = np.zeros((n_steps, 2))

accelerations_1 = np.zeros((n_steps, 2))
accelerations_2 = np.zeros((n_steps, 2))

sim_copy = sim.copy()
for i, t in enumerate(times):
    sim_copy.integrate(t)
    p0, p1, p2 = sim_copy.particles
    positions_0[i] = [p0.x, p0.y]
    positions_1[i] = [p1.x, p1.y]
    positions_2[i] = [p2.x, p2.y]

    accelerations_1[i] = [p1.ax, p1.ay]
    accelerations_2[i] = [p2.ax, p2.ay]

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

# Compute Earth Angle Around the Sun
earth_angle = np.zeros(n_steps)
for i in range(n_steps):
    earth_vec = positions_1[i] - positions_0[i]
    sidereal_vec = [1, 0]
    if np.linalg.norm(earth_vec) == 0:
        earth_angle[i] = np.nan
    else:
        earth_vec_norm = earth_vec / np.linalg.norm(earth_vec)
        earth_angle[i] = np.dot(earth_vec, sidereal_vec)

# Earth New Year Detection
earth_peaks, _ = find_peaks(earth_angle)
new_year = times[earth_peaks]
earth_periods = np.array([b - a for a, b in zip(new_year[:-1], new_year[1:])])

# Peak Detection
ratio_peaks, _ = find_peaks(ratio_vals,height=1)
peak_time = times[ratio_peaks]

# Synodic Period Histogram 
synodic_periods = np.array([b - a for a, b in zip(peak_time[:-1], peak_time[1:])])
syn_per_counts, bin_edges = np.histogram(synodic_periods, bins=num_bins)

# Mars Period 
mars_periods = (earth_periods*synodic_periods) / (synodic_periods - earth_periods)
m_per_counts, bin_edges = np.histogram(mars_periods, bins=num_bins)

# Mars Radius / Angular Velocity 
mars_radii = np.cbrt((mars_periods**2 * G * central_mass) / (2*np.pi**2))
m_rad_counts, bin_edges = np.histogram(mars_radii, bins=num_bins)

mars_angular_velocities = 2*np.pi / mars_periods
m_ang_vel_counts, bin_edges = np.histogram(mars_angular_velocities, bins=num_bins)

# === Plot Setup ===
fig, ((ax_orbit, ax_synodic), (ax_mars, ax_radius)) = plt.subplots(2, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25, hspace=0.35)
idx = 0 # Animation starts on this frame

# 1️⃣ Orbit Plot
# ax_orbit.set_xlim(-1.6, 1.6)
# ax_orbit.set_ylim(-1.6, 1.6)
ax_orbit.set_aspect('equal')
ax_orbit.set_title("Orbital Motion with Normalized Perturbation and Earth–Mars Vectors")
ax_orbit.grid(True)

ax_orbit.plot(positions_1[:, 0], positions_1[:, 1], 'b-', alpha=0.3)
ax_orbit.plot(positions_2[:, 0], positions_2[:, 1], 'r-', alpha=0.3)
ax_orbit.plot(positions_0[:, 0], positions_0[:, 1], 'y-', alpha=0.3)

marker_sun, = ax_orbit.plot([], [], 'yo', markersize=8)
marker_earth, = ax_orbit.plot([], [], 'bo', markersize=5)
marker_mars, = ax_orbit.plot([], [], 'ro', markersize=5)
c_marker_mars, = ax_orbit.plot([], [], "o", color="red", alpha=0.5, markersize = 6) # calculated position of mars based off measurements
pert_arrow = None
mars_arrow = None
ax_orbit.legend(loc="upper right")


# Synodic Period Histogram Plot 
ax_synodic.bar(bin_edges[:-1], syn_per_counts, width=np.diff(bin_edges), align='edge', edgecolor='black')
ax_synodic.set_title("PDF of Synodic Period")
ax_synodic.set_ylabel("Number of Occurances")
ax_synodic.set_xlabel("Years")

# Mars Period Histogram Plot 
ax_mars.bar(bin_edges[:-1], m_per_counts, width=np.diff(bin_edges), align='edge', edgecolor='black')
ax_mars.set_title("PDF of Mars Period")
ax_mars.set_ylabel("Number of Occurances")
ax_mars.set_xlabel("Years")

# Mars Radius Histogram Plot 
ax_radius.bar(bin_edges[:-1], m_rad_counts, width=np.diff(bin_edges), align='edge', edgecolor='black')
ax_radius.set_title("PDF of Mars Radius")
ax_radius.set_ylabel("Number of Occurances")
ax_radius.set_xlabel("AU")


# # 2️⃣ Combined Cosine & Ratio Plot
# ax_combined.set_xlim(times[0], times[-1])
# ax_combined.set_title("Cosine (Magenta) and Accel Ratio (Green)")
# ax_combined.set_xlabel("Time (s)")

# # Left y-axis for cosine
# ax_combined.set_ylabel("Cosine", color='m')
# ax_combined.set_ylim(-1.1, 1.1)
# corr_line, = ax_combined.plot([], [], 'm-', label="Cosine (pert vs Mars)")

# # Right y-axis for ratio
# ax_ratio_twin = ax_combined.twinx()
# ax_ratio_twin.set_ylabel("Accel Ratio |Sun|/|Net|", color='g')
# # ax_ratio_twin.set_ylim(0.9*np.nanmin(ratio_vals), 1.1*np.nanmax(ratio_vals))
# ratio_line, = ax_ratio_twin.plot([], [], 'g-', label="Accel Ratio")

# # Find points of interest in ratio plot
# ratio_peaks, _ = find_peaks(ratio_vals,height=1)
# peak_time = times[ratio_peaks]
# peak_height = ratio_vals[ratio_peaks]
# ax_ratio_twin.scatter(peak_time, peak_height)

# # Vertical time marker
# corr_time_marker = ax_combined.axvline(times[0], color='k', ls='--')

# === Update Function ===
def update(i):

    global pert_arrow, mars_arrow, offset, detected_peaks, T1, T2, c_r2
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

    # corr_line.set_data(times[:i], corr_vals[:i])
    # ratio_line.set_data(times[:i], (ratio_vals[:i]-1)+1)
    # corr_time_marker.set_xdata([times[i], times[i]])

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
