import rebound
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(central_mass, G=6.67430e-11, t_max=1e6, n_steps=1e4):
    """Run the simulation for a given central_mass and return the final cumulative correlation."""
    sim = rebound.Simulation()
    sim.units = ('s', 'm', 'kg')
    sim.integrator = "ias15"

    # Add central mass and two orbiting planets
    sim.add(m=central_mass)  # central star (index 0)

    r1, r2 = 1.0, 2
    m1 = m2 = 1.0
    v1 = np.sqrt(G * central_mass / r1)
    v2 = np.sqrt(G * central_mass / r2)

    sim.add(m=m1, x=r1, y=0, vy=v1)  # Earth-like
    sim.add(m=m2, x=r2, y=0, vy=v2)  # Mars-like

    # Time grid
    times = np.linspace(0, t_max, int(n_steps))
    positions_0 = np.zeros((len(times), 2))
    positions_1 = np.zeros((len(times), 2))
    positions_2 = np.zeros((len(times), 2))

    sim_copy = sim.copy()
    for i, t in enumerate(times):
        sim_copy.integrate(t)
        p0, p1, p2 = sim_copy.particles
        positions_0[i] = (p0.x, p0.y)
        positions_1[i] = (p1.x, p1.y)
        positions_2[i] = (p2.x, p2.y)

    # === Compute accelerations ===
    dt = times[1] - times[0]
    accel_earth_approx = np.zeros_like(positions_1)
    for i in range(2, len(times)):
        accel_earth_approx[i] = (positions_1[i] - 2 * positions_1[i-1] + positions_1[i-2]) / dt**2

    accel_sun = np.zeros_like(positions_1)
    for i in range(len(times)):
        r_vec = positions_1[i] - positions_0[i]
        r = np.linalg.norm(r_vec)
        if r > 0:
            accel_sun[i] = -G * central_mass * r_vec / r**3

    # === Perturbation acceleration ===
    accel_mars = accel_earth_approx - accel_sun

    # === Compute correlation ===
    corr_values = []
    for i in range(len(times)):
        pert_vec = accel_mars[i]
        to_mars_vec = positions_2[i] - positions_1[i]

        norm_pert = np.linalg.norm(pert_vec)
        norm_to_mars = np.linalg.norm(to_mars_vec)
        if norm_pert == 0 or norm_to_mars == 0:
            continue

        pert_unit = pert_vec / norm_pert
        to_mars_unit = to_mars_vec / norm_to_mars

        cosine_corr = np.dot(pert_unit, to_mars_unit)
        corr_values.append(cosine_corr)

    return np.mean(corr_values) if len(corr_values) > 0 else np.nan


# === Parameter sweep over central_mass ===
central_mass_values = np.linspace(1, 400, 100)
correlations = []

print("Running simulations...")

for cm in central_mass_values:
    corr = run_simulation(central_mass=cm)
    correlations.append(corr)
    print(f"Central mass = {cm:.1f}, correlation = {corr:.6f}")

# === Plot results ===
plt.figure(figsize=(8, 5))
plt.plot(central_mass_values, correlations, 'o-', color='purple')
plt.xlabel("Central Mass")
plt.ylabel("Cumulative Correlation (mean cosine of angle)")
plt.title("Correlation Between Perturbation and Earth→Mars Vector vs. Central Mass")
plt.grid(True)
plt.tight_layout()
plt.show()