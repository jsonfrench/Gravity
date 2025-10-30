# OrbitalPlots_testing file

import rebound
import numpy as np
from orbitalplots import OrbitalPlots

# ============================================================
# =============== Build Sun–Earth–Mars system =================
# ============================================================

# --- Physical constants ---
G = 6.67430e-11  # gravitational constant (m^3 / kg / s^2)
mass_sun = 1.9885e30
mass_mercury = 3.301e23
mass_venus = 4.867e24
mass_earth = 5.972e24
mass_mars = 6.4171e23
mass_jupiter = 1.898e27
mass_saturn = 5.683e26

# --- Orbital radii (m) ---
r_mercury = 5.79e10
r_venus = 1.08e11   # 0.728 au
r_earth = 1.496e11  # 1 au
r_mars = 2.279e11   # 1.524 au
r_jupiter = 7.786e11
r_saturn = 1.43e12

# --- Circular velocities (m/s) assuming central mass = Sun ---
v_mercury = np.sqrt(G * mass_sun / r_mercury)
v_venus = np.sqrt(G * mass_sun / r_venus)
v_earth = np.sqrt(G * mass_sun / r_earth)
v_mars = np.sqrt(G * mass_sun / r_mars)
v_jupiter = np.sqrt(G * mass_sun / r_jupiter)
v_saturn = np.sqrt(G * mass_sun / r_saturn)

# --- Create simulation ---
sim = rebound.Simulation()
sim.units = ('s', 'm', 'kg')
sim.integrator = "ias15"

# Add bodies -- for ratio computations to make sense later, this
# should be done in SUN - EARTH - MARS - OTHERS order
sim.add(m=mass_sun)                                 # Sun
sim.add(m=mass_earth, x=r_earth, y=0, vy=v_earth)   # Earth
sim.add(m=mass_jupiter, x=r_jupiter, y=0, vy=v_jupiter) # jupiter being third makes it the unknown
sim.add(m=mass_mars, x=r_mars, y=0, vy=v_mars)      # Mars
sim.add(m=mass_venus, x = r_venus, y=0, vy=v_venus) # Venus
# sim.add(m=mass_mercury, x = r_mercury, y=0, vy=v_mercury)
# sim.add(m=mass_saturn, x=r_saturn, y=0, vy=v_saturn)

# --- Time setup ---
YEARS = 30
t_max = YEARS * 365.25 * 24 * 60 * 60  
n_steps = int(t_max // 3600)
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

# initialize a net acceleration array
net_accel_others = np.zeros_like(positions[1])

# create boolean flag to indicate other known bodies are present
other_bodies_flag = False
if len(positions) > 3:
    # switch flag to indicate other known bodies are present
    other_bodies_flag = True
    
    # if the system contains more known bodies than S-E-M, create array of 0s with the shape 
    # (# other bodies = # total bodies - 3 to acct for S-E-M, length of earth orbital array)
    # for storing the accelerations on earth due to other bodies, then create an array
    # with shape (# other bodies, n_steps, 2) for storing the vectors pointing from 
    # Earth toward the other known bodies, as well as an array for the magnitudes of those vectors 
    accel_others = np.zeros((len(positions)-3, len(positions[1]), 2))
    r_earth_others = np.zeros((len(positions)-3, n_steps, 2))
    mag_r_earth_others = np.zeros((len(positions)-3, n_steps))
for i in range(n_steps):
    r_sun_earth = positions[1][i] - positions[0][i]
    if other_bodies_flag:
        for j in range(len(positions)-3):
            r_earth_others[j, i] = positions[1][i] - positions[j+3][i]
            mag_r_earth_others[j, i] = np.linalg.norm(r_earth_others[j, i])
            if mag_r_earth_others[j, i] > 0:
                # cheating here and using the mass of venus.  We should probably store masses
                # in an array at the beginning?
                accel_others[j, i] = -G * mass_venus * r_earth_others[j, i] / mag_r_earth_others[j, i]**3

    mag_r_sun_earth = np.linalg.norm(r_sun_earth)
    
    # compute acceleration on earth due to sun
    if mag_r_sun_earth > 0:
        accel_sun[i] = -G * mass_sun * r_sun_earth / mag_r_sun_earth**3
    
    # compute net acceleration due to known bodies at each time
    if other_bodies_flag:
        for j in range(len(positions)-3):
            net_accel_others[i] = net_accel_others[i] + accel_others[j, i]
        
    net_accel_others[i] = net_accel_others[i] + accel_sun[i]



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
    ratio_vals[i] = mag_others / mag_net if mag_net > 0 else ratio_vals[i-1] if i>0 else 1

ratio_vals = np.delete(ratio_vals, [0,len(ratio_vals)-1])

# accentuate the fluctuations in the ratio array -- tweak procedurally for more known bodies?
# for instance [sum of known non-earth masses]/[earth mass]?
ratio_vals = (ratio_vals - 1) * (mass_sun / mass_earth) + 1

corr_vals = np.zeros(n_steps)
for i in range(n_steps):
    a = accel_mars[i]
    b = positions[2][i] - positions[1][i]
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        corr_vals[i] = np.nan
    else:
        corr_vals[i] = np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))

corr_vals = np.delete(corr_vals, [0, len(corr_vals)-1])
# ============================================================
# =============== Launch OrbitalPlots visualization ==========
# ============================================================

times_years = times / (365.25 * 24 * 3600)  # convert seconds to years

lim = compute_plot_limits([r_earth,r_mars])

plots = OrbitalPlots(
    positions_list=positions,
    times_years=times_years,
    ratio_vals=ratio_vals,
    corr_vals=corr_vals,
    xlim=lim,
    ylim=lim,
    mov_avg_len=19,
    prominence_val=0.05
)

# plots.create_ratio_cosine_figure()

# plots.create_orbit_figure()

# plots.create_ratio_cosine_figure()
plots.plot_ratio_cosine_with_synodic_fft()
plots.show_plots()

# plots.plot_fft()
