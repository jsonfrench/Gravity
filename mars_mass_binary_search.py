"""
sindy_mars_mass_binary_search.py

Standalone 2D SEM pipeline to estimate Mars' mass using short-term
REBOUND simulations and a binary search, based on Earth's orbital perturbation.

Units: AU, yr, Msun
"""

import numpy as np
import rebound
import math
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

# ---------------------------
# Physical constants & units
# ---------------------------
G = 4*np.pi**2  # AU^3 / (Msun yr^2)
AU = 1.0
M_sun = 1.0
M_earth = 3.003e-6  # Earth in Msun
mass_factor = 1
M_mars_true = mass_factor * 3.227e-7  # Mars in Msun, for validation

# ---------------------------
# Simulation parameters
# ---------------------------
YEARS = 20
dt = 1/(365.25)  # 1 day in years
times = np.arange(0, YEARS + dt, dt)
N = len(times)

r_earth = 1.0  # AU
r_mars_true = 1.523679  # AU

# Circular velocities
v_factor = 1
v_earth = np.sqrt(G*M_sun/r_earth)
v_mars_true = np.sqrt(G*M_sun/r_mars_true) * v_factor

theta_e0 = 0.0
theta_m0 = math.radians(60)

# ---------------------------
# Function: simulate SEM system
# ---------------------------
def simulate_sem(mars_mass, r_mars=r_mars_true, v_mars=None, times=times):
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = "ias15"
    sim.dt = dt

    # Sun
    sim.add(m=M_sun, x=0, y=0, z=0, vx=0, vy=0, vz=0, hash="sun")

    # Earth
    sim.add(
        m=M_earth,
        x=r_earth*math.cos(theta_e0),
        y=r_earth*math.sin(theta_e0),
        z=0,
        vx=-v_earth*math.sin(theta_e0),
        vy=v_earth*math.cos(theta_e0),
        vz=0,
        hash="earth"
    )

    # Mars
    if v_mars is None:
        v_mars = math.sqrt(G*M_sun/r_mars)
    sim.add(
        m=mars_mass,
        x=r_mars*math.cos(theta_m0),
        y=r_mars*math.sin(theta_m0),
        z=0,
        vx=-v_mars*math.sin(theta_m0),
        vy=v_mars*math.cos(theta_m0),
        vz=0,
        hash="mars"
    )

    sim.move_to_com()

    earth_pos = np.zeros((len(times),2))
    mars_pos = np.zeros((len(times),2))

    for i,t in enumerate(times):
        sim.integrate(t)
        e = sim.particles["earth"]
        m = sim.particles["mars"]
        earth_pos[i] = [e.x, e.y]
        mars_pos[i] = [m.x, m.y]

    return earth_pos, mars_pos

# ---------------------------
# Function: compute acceleration ratio
# ---------------------------
def central_second_order_acc(pos, dt):
    Np = len(pos)
    acc = np.zeros_like(pos)
    for i in range(1, Np-1):
        acc[i] = (pos[i+1]-2*pos[i]+pos[i-1])/(dt**2)
    acc[0] = (pos[2]-2*pos[1]+pos[0])/(dt**2)
    acc[-1] = (pos[-1]-2*pos[-2]+pos[-3])/(dt**2)
    return acc

def sun_acc(pos):
    r = np.linalg.norm(pos, axis=1)
    r_safe = np.where(r==0,1e-12,r)
    return -(G*M_sun)*(pos.T/r_safe**3).T

def compute_ratio(earth_pos):
    a_net = central_second_order_acc(earth_pos, dt)
    a_s = sun_acc(earth_pos)
    ratio = np.linalg.norm(a_s,axis=1)/np.linalg.norm(a_net,axis=1)
    return ratio

# ---------------------------
# Step 1: Long-term sim for ratio array
# ---------------------------
print("Running long-term SEM simulation for synodic period...")
earth_pos_long, mars_pos_long = simulate_sem(M_mars_true, r_mars_true, v_mars=v_mars_true)

ratio_long = compute_ratio(earth_pos_long)

# Smooth ratio
def moving_average(x,w):
    return np.convolve(x, np.ones(w)/w, mode='valid')
MA_len = 11
ratio_smooth = moving_average(ratio_long, MA_len)
times_smooth = times[(MA_len-1)//2:-(MA_len//2)] if MA_len%2==1 else times[MA_len-1:]

# ---------------------------
# Step 2: Estimate synodic period
# ---------------------------
peaks_idx,_ = find_peaks(ratio_smooth, height=None, distance=1)
if len(peaks_idx)<2:
    # fallback FFT
    Nfft = len(ratio_smooth)
    dt_years = times_smooth[1]-times_smooth[0]
    freqs = rfftfreq(Nfft, dt_years)
    fft_mag = np.abs(rfft(ratio_smooth-np.mean(ratio_smooth)))
    mask = (freqs>0) & (freqs<=1.0)
    if np.any(mask):
        idx_peak = np.argmax(fft_mag[mask])
        freq_peak = freqs[mask][idx_peak]
        synodic_years = 1.0/freq_peak
    else:
        raise RuntimeError("Unable to estimate synodic period")
else:
    peak_times = times_smooth[peaks_idx]
    diffs = np.diff(peak_times)
    synodic_years = np.median(diffs)

print(f"Synodic period ≈ {synodic_years:.4f} yr")

# Mars orbital period
P_earth = 1.0
inv = 1/P_earth - 1/synodic_years
P_mars = 1/inv if inv>0 else 1.8808
print(f"Predicted Mars period: {P_mars:.4f} yr")

# Predicted radius (Kepler)
a_pred = (G*M_sun*P_mars**2/(4*np.pi**2))**(1/3)
print(f"Predicted Mars semi-major axis: {a_pred:.4f} AU")


def compare_earth_final_positions(r_mars, v_mars, M_mars_true, times_short, dt=1/365.25, zoom_factor=1.1):
    """
    Compute short-term Earth orbits with Mars mass: true, doubled, halved.
    Plot a zoomed-in view around the final Earth positions.
    
    Parameters
    ----------
    r_mars : float
        Mars orbital radius (AU)
    v_mars : float
        Mars circular velocity (AU/yr)
    M_mars_true : float
        Mars mass (Msun)
    times_short : ndarray
        Array of times to integrate over (yr)
    dt : float
        Timestep in years
    zoom_factor : float
        Factor to expand the zoomed-in view around the final positions
    """
    theta_e0 = 0.0
    theta_m0 = math.radians(60)
    r_earth = 1.0
    v_earth = math.sqrt(4*np.pi**2 / r_earth)  # AU/yr
    M_sun = 1.0
    M_earth = 3.003e-6

    def run_sim(M_mars):
        sim = rebound.Simulation()
        sim.units = ('AU','yr','Msun')
        sim.integrator = "ias15"
        sim.dt = dt

        sim.add(m=M_sun, x=0, y=0, z=0, vx=0, vy=0, vz=0, hash="sun")
        sim.add(m=M_earth, x=r_earth*math.cos(theta_e0), y=r_earth*math.sin(theta_e0), z=0,
                vx=-v_earth*math.sin(theta_e0), vy=v_earth*math.cos(theta_e0), vz=0, hash="earth")
        sim.add(m=M_mars, x=r_mars*math.cos(theta_m0), y=r_mars*math.sin(theta_m0), z=0,
                vx=-v_mars*math.sin(theta_m0), vy=v_mars*math.cos(theta_m0), vz=0, hash="mars")

        sim.move_to_com()
        earth_pos = np.zeros((len(times_short),2))
        for i,t in enumerate(times_short):
            sim.integrate(t)
            e = sim.particles["earth"]
            earth_pos[i] = [e.x, e.y]
        return earth_pos

    # Compute trajectories
    big = 2
    small = 0.5
    earth_true = run_sim(M_mars_true)
    earth_double = run_sim(big * M_mars_true)
    earth_half = run_sim(small * M_mars_true)

    # Determine zoom window
    final_positions = np.vstack([earth_true[-1], earth_double[-1], earth_half[-1]])
    x_min, y_min = final_positions.min(axis=0)
    x_max, y_max = final_positions.max(axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_range = (x_max - x_min) * zoom_factor
    y_range = (y_max - y_min) * zoom_factor

    print(f"\nEARTH R DIFF = {np.linalg.norm(earth_true[-1])- np.linalg.norm(earth_half[-1])}")

    # Plot final positions zoomed in
    plt.figure(figsize=(6,6))
    plt.scatter(earth_true[-1,0], earth_true[-1,1], color='blue', label="Earth (Mars mass true)")
    plt.scatter(earth_double[-1,0], earth_double[-1,1], color='red', label="Earth (Mars mass doubled)")
    plt.scatter(earth_half[-1,0], earth_half[-1,1], color='green', label="Earth (Mars mass halved)")
    plt.scatter([0],[0], color='orange', label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlim(x_center - x_range/2, x_center + x_range/2)
    plt.ylim(y_center - y_range/2, y_center + y_range/2)
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("Zoomed-in final Earth positions for different Mars masses")
    plt.legend()
    plt.show()


t_short = 1/12  # 1 month
times_short = np.linspace(0, t_short, int(t_short/dt))
r_mars_pred = 1.523679  # AU
v_mars_pred = math.sqrt(G / r_mars_pred)

# compare_earth_final_positions(r_mars_pred, v_mars_pred, M_mars_true, times_short, dt=dt)




# ---------------------------
# Step 3-7: Short-term sim loop + binary search for Mars mass
# ---------------------------
t_short = 1/52  
N_short = int(t_short/dt)
max_iters = 1000
tol = 1e-16  # AU

mars_guess = M_earth  # initial guess
mars_guess_prev = 0.0

print("\nStarting binary search for Mars mass...")

for iteration in range(max_iters):
    v_mars_guess = 2 * np.pi / np.sqrt(a_pred) * v_factor
    # simulate short-term with current guess
    earth_pos_short, mars_pos_short = simulate_sem(mars_guess, r_mars=a_pred, v_mars=v_mars_guess, times=np.linspace(0,t_short,N_short))
    earth_final_sim = earth_pos_short[-1]
    earth_final_true = earth_pos_long[N_short-1]

    diff = np.linalg.norm(earth_final_sim) - np.linalg.norm(earth_final_true)

    print(f"Iter {iteration}: Mars guess = {mars_guess:.6e} Msun, |Δr| = {diff:.6e}")

    if np.abs(diff) < tol:
        print("Converged!")
        break

    # Update guess: simple binary search
    if diff>0: # check if simmed earth is farther away than expected --> mars' mass is too small
        mars_guess_prev = mars_guess
        mars_guess *= 2
    else:
        # Mars mass too big
        if mars_guess_prev==0:
            mars_guess /= 2
        else:
            mars_guess = 0.5*(mars_guess + mars_guess_prev)
    # else:
    #     # should rarely happen
    #     break

print(f"\nEstimated Mars mass ≈ {mars_guess:.6e} Msun")
print(f"True Mars mass = {M_mars_true:.6e} Msun")
print(f"Ratio = {mars_guess/M_mars_true:.6f}")

# ---------------------------
# Optional: plot Earth/Mars short-term comparison
# ---------------------------
plt.figure(figsize=(6,6))
plt.plot(earth_pos_long[:N_short,0], earth_pos_long[:N_short,1], label='Earth (true)')
plt.plot(earth_pos_short[:,0], earth_pos_short[:,1], '--', label='Earth (sim guess)')
plt.scatter([0],[0], color='orange', label='Sun')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Earth orbit comparison (short-term)")
plt.show()
