import rebound
import numpy as np
from Test_Class import OrbitalPlots

# ============================================================
# =============== Build Sun–Earth–Mars system =================
# ============================================================

# --- Physical constants ---
G = 6.67430e-11  # gravitational constant (m^3 / kg / s^2)
mass_sun = 1.9885e30
mass_venus = 4.867e24
mass_earth = 5.972e24
mass_mars = 6.4171e23

# --- Orbital radii (m) ---
r_venus = 1.08e11   # 0.728 au
r_earth = 1.496e11  # 1 au
r_mars = 2.279e11   # 1.524 au

# --- Circular velocities (m/s) assuming central mass = Sun ---
v_venus = np.sqrt(G * mass_sun / r_venus)
v_earth = np.sqrt(G * mass_sun / r_earth)
v_mars = np.sqrt(G * mass_sun / r_mars)

# --- Create simulation ---
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

# Add bodies -- for ratio computations to make sense later, this
# should be done in SUN - EARTH - MARS - OTHERS order
sim.add(m=mass_sun)                                 # Sun
sim.add(m=mass_earth, x=r_earth, y=0, vy=v_earth)   # Earth
sim.add(m=mass_mars, x=r_mars, y=0, vy=v_mars)      # Mars
# sim.add(m=mass_venus, x = r_venus, y=0, vy=v_venus) # Venus

# --- Time setup ---
year = 365.25 * 24 * 3600  # total time in seconds
t_max = 10 * year    
n_steps = int(t_max / 3600) # one step per hour
times = np.linspace(0, t_max, n_steps)
dt = times[1] - times[0]

# ============================================================
# =============== Integrate and record positions =============
# ============================================================
positions = [np.zeros((n_steps, 2)) for _ in range(len(sim.particles))]

sim_copy = sim.copy()
for i, t in enumerate(times):
    sim_copy.integrate(t)
    for j, p in enumerate(sim_copy.particles):
        positions[j][i] = [p.x, p.y]

# ============================================================
# =============== Compute accelerations ======================
# ============================================================

def finite_diff_accel(pos, dt):
    """Second-order finite difference acceleration."""
    acc = np.zeros_like(pos)
    for k in range(2, len(pos)):
        acc[k-1] = (pos[k] - 2*pos[k-1] + pos[k-2]) / dt**2
    return acc

# method to compute plotting limits if using eccentric orbits
# pass it an array containing the orbital radii for all bodies and
# a desired eccentricity value (defaults to 0)
def compute_plot_limits(R, ecc=0):
    return 2 * np.max(R) / (1 - ecc**2)

accel_earth_approx = finite_diff_accel(positions[1], dt)

# Gravitational acceleration due to Sun and any other bodies
accel_sun = np.zeros_like(positions[1])

# --- Setup for other bodies ---
other_bodies_flag = len(positions) > 3

# Always allocate arrays so names exist even if there are no 'other' bodies
net_accel_others = np.zeros_like(positions[1])  # shape (n_steps, 2)
accel_others = None
r_earth_others = None
mag_r_earth_others = None

if other_bodies_flag:
    n_other = len(positions) - 3  # number of bodies beyond Sun–Earth–Mars
    accel_others = np.zeros((n_other, n_steps, 2))
    r_earth_others = np.zeros((n_other, n_steps, 2))
    mag_r_earth_others = np.zeros((n_other, n_steps))

# --- Time integration loop ---
for i in range(n_steps):
    r_sun_earth = positions[1][i] - positions[0][i]
    mag_r_sun_earth = np.linalg.norm(r_sun_earth)

    # Acceleration on Earth due to Sun
    if mag_r_sun_earth > 0:
        accel_sun[i] = -G * mass_sun * r_sun_earth / mag_r_sun_earth**3
    else:
        accel_sun[i] = 0.0

    # Start the net-known acceleration with the Sun's contribution
    net_accel_others[i] = accel_sun[i]

    # Add contributions from any additional known bodies
    if other_bodies_flag:
        for j in range(n_other):
            r_earth_others[j, i] = positions[1][i] - positions[j+3][i]
            mag_r_earth_others[j, i] = np.linalg.norm(r_earth_others[j, i])
            if mag_r_earth_others[j, i] > 0:
                # TODO: replace mass_venus with the actual mass of the j-th body
                accel_others[j, i] = -G * mass_venus * r_earth_others[j, i] / mag_r_earth_others[j, i]**3
            else:
                accel_others[j, i] = 0.0

            net_accel_others[i] += accel_others[j, i]


# Perturbing acceleration due to Mars
# accel_mars = accel_earth_approx - accel_sun

accel_mars = accel_earth_approx - net_accel_others


# ============================================================
# =============== Compute ratio and cosine arrays ============
# ============================================================

# NOTE: these ratios need to be procedurally generated to include the known
# influence of every known body on earth, not just the sun.  The numerator should
# be |sum of all known accelerations acting on earth|
ratio_vals = np.zeros(n_steps)
for i in range(n_steps):
    # mag_sun = np.linalg.norm(accel_sun[i])
    mag_others = np.linalg.norm(net_accel_others[i])
    mag_net = np.linalg.norm(accel_earth_approx[i])
    ratio_vals[i] = mag_others / mag_net if mag_net > 0 else np.nan

# accentuate the fluctuations in the ratio array -- tweak procedurally for more known bodies?
ratio_vals = (ratio_vals - 1) * (mass_sun / mass_earth) + 1

corr_vals = np.zeros(n_steps)
for i in range(n_steps):
    a = accel_mars[i]
    b = positions[2][i] - positions[1][i]
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        corr_vals[i] = np.nan
    else:
        corr_vals[i] = np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))

# ============================================================
# =============== Launch OrbitalPlots visualization ==========
# ============================================================

times_years = times / (365.25 * 24 * 3600)  # convert seconds to years

plots = OrbitalPlots(
    positions_list=positions,
    ratio_vals=ratio_vals,
    corr_vals=corr_vals,
    times_years=times_years,
    xlim=2.5e11,
    ylim=2.5e11,
    mov_avg_len=19,
    prominence_val=0.05
)


# ============================================================
# ============== Compute and print peak statistics ===========
# ============================================================

mu = G*mass_sun
res = plots.peak_stats_and_kepler(which='ratio', unit='days', mu=mu, sun_idx=0, earth_idx=1)
r_mars = plots.mars_distance_from_period(mu=mu, period_days=res['mars_T_from_synodic_days'])
print("------------ Earth & Mars Periods ------------")
print(f"Synodic Period Approximation: {res['median']:.2f} {res['unit']}")
print(f"Earth period (Kepler, a≈⟨r⟩): {res['earth_T_kepler_days']:.2f} days ({res['earth_T_kepler_years']:.6f} years)")
print(f"Mars period (from synodic): {res['mars_T_from_synodic_days']:.2f} days ({res['mars_T_from_synodic_years']:.6f} years)")
print("------------ Mars Orbital Radius (Kepler Inverse) ------------")
print(f"Estimated Mars distance from Sun ≈ {r_mars:.3e} m ({r_mars/1.496e11:.3f} AU)")

# plots.mars_ref_radius = plots.mars_distance_from_period(mu=mu, period_days=res['mars_T_from_synodic_days'])

plots.create_ratio_cosine_figure()

# plots.create_orbit_figure()


