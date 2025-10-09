import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# === Setup Simulation ===
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

G = 6.67430e-11  # gravitational constant

# Contrived Sun–Earth–Mars system
central_mass = 10.0
sim.add(m=central_mass)   # Sun
r1, r2 = 1.0, 1.2         # orbital radii
m1 = m2 = 1.0
v1 = np.sqrt(G * central_mass / r1)
v2 = np.sqrt(G * central_mass / r2)
sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth
sim.add(m=m2, x=r2, y=0, vy=v2)  # Mars

# === Time Setup ===
t_max = 2e6
n_steps = 3000
times = np.linspace(0, t_max, n_steps)

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

# Perturbation (acceleration due to Mars)
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
        corr_vals[i] = np.dot(a_norm, b_norm)  # cosine of angle

    mag_accel_sun = np.linalg.norm(accel_sun[i])
    mag_accel_net = np.linalg.norm(accel_earth_approx[i])
    ratio_vals[i] = mag_accel_sun / mag_accel_net if mag_accel_net > 0 else np.nan

# === Plot Setup ===
fig, (ax_orbit, ax_corr, ax_ratio) = plt.subplots(3, 1, figsize=(6, 10))
plt.subplots_adjust(bottom=0.25, hspace=0.35)

# 1️⃣ Orbit Plot
ax_orbit.set_xlim(-1.6, 1.6)
ax_orbit.set_ylim(-1.6, 1.6)
ax_orbit.set_aspect('equal')
ax_orbit.set_title("Orbital Motion with Normalized Perturbation and Earth–Mars Vectors")
ax_orbit.grid(True)

# Static orbit traces
ax_orbit.plot(positions_1[:, 0], positions_1[:, 1], 'b-', alpha=0.3, label='Earth Orbit')
ax_orbit.plot(positions_2[:, 0], positions_2[:, 1], 'r-', alpha=0.3, label='Mars Orbit')
ax_orbit.plot(positions_0[:, 0], positions_0[:, 1], 'yo', label='Sun')

marker_sun, = ax_orbit.plot([], [], 'yo', markersize=5)
marker_earth, = ax_orbit.plot([], [], 'bo', markersize=5)
marker_mars, = ax_orbit.plot([], [], 'ro', markersize=5)
pert_arrow = None
mars_arrow = None
ax_orbit.legend()

# 2️⃣ Correlation Plot
ax_corr.set_xlim(times[0], times[-1])
ax_corr.set_ylim(-1.1, 1.1)
ax_corr.set_title("Cosine of Angle Between Perturbation and Earth–Mars Vector")
ax_corr.set_xlabel("Time (s)")
ax_corr.set_ylabel("Cosine")
corr_line, = ax_corr.plot([], [], 'm-')
corr_time_marker = ax_corr.axvline(times[0], color='k', ls='--')

# 3️⃣ Ratio Plot
ax_ratio.set_xlim(times[0], times[-1])
ax_ratio.set_ylim(0.9 * np.nanmin(ratio_vals), 1.1 * np.nanmax(ratio_vals))
ax_ratio.set_title("Ratio of |Accel_sun| / |Accel_net|")
ax_ratio.set_xlabel("Time (s)")
ax_ratio.set_ylabel("Ratio")
ratio_line, = ax_ratio.plot([], [], 'g-')
ratio_time_marker = ax_ratio.axvline(times[0], color='k', ls='--')

# === Update Function ===
def update(i):
    global pert_arrow, mars_arrow

    # Update moving markers
    marker_sun.set_data([positions_0[i, 0]], [positions_0[i, 1]])
    marker_earth.set_data([positions_1[i, 0]], [positions_1[i, 1]])
    marker_mars.set_data([positions_2[i, 0]], [positions_2[i, 1]])

    # Remove previous arrows
    for arrow in [pert_arrow, mars_arrow]:
        if arrow is not None:
            arrow.remove()
    pert_arrow = mars_arrow = None

    # Draw normalized vectors at Earth's position
    earth_pos = positions_1[i]
    pert_vec = accel_mars[i]
    mars_vec = positions_2[i] - positions_1[i]

    if np.linalg.norm(pert_vec) > 0 and np.linalg.norm(mars_vec) > 0:
        pert_unit = pert_vec / np.linalg.norm(pert_vec)
        mars_unit = mars_vec / np.linalg.norm(mars_vec)
        arrow_len = 0.3  # fixed length for normalized vectors
        pert_arrow = ax_orbit.arrow(earth_pos[0], earth_pos[1],
                                    pert_unit[0]*arrow_len, pert_unit[1]*arrow_len,
                                    color='magenta', head_width=0.03)
        mars_arrow = ax_orbit.arrow(earth_pos[0], earth_pos[1],
                                    mars_unit[0]*arrow_len, mars_unit[1]*arrow_len,
                                    color='cyan', head_width=0.03)

    # Update correlation and ratio plots
    corr_line.set_data(times[:i], corr_vals[:i])
    ratio_line.set_data(times[:i], ratio_vals[:i])
    corr_time_marker.set_xdata([times[i], times[i]])
    ratio_time_marker.set_xdata([times[i], times[i]])

# === Slider and Button ===
ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
slider = Slider(ax_slider, 'Time', 0, n_steps - 1, valinit=0, valstep=1)

paused = False
def toggle(event):
    global paused
    paused = not paused

ax_button = plt.axes([0.82, 0.11, 0.1, 0.04])
button = Button(ax_button, 'Play/Pause')
button.on_clicked(toggle)

def slider_update(val):
    i = int(slider.val)
    update(i)
    fig.canvas.draw_idle()
slider.on_changed(slider_update)

def animate(frame):
    if not paused:
        slider.set_val(frame)
    return []

ani = FuncAnimation(fig, animate, frames=n_steps, interval=30, blit=False, repeat=True)

plt.show()
