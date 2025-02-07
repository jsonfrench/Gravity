# Jason French 1.15.24

import numpy as np
from scipy.integrate import odeint
from scipy.fft import fft
from scipy.fft import rfft
import matplotlib.pyplot as plt


# Constatnts
G = 6.674e-11

# Simulation setup
samples_per_time = 10
time_steps = 100000 * 1
t = np.linspace(0, time_steps, time_steps*samples_per_time) 
freq = np.fft.rfftfreq(len(t), 1/samples_per_time) * 10

def one_body(IVP, t, G, M): 
    
    x, y, vx, vy = IVP

    r = np.sqrt(x**2 + y**2)

    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

def two_body(IVP, t, G, M, m1, m2): 
    
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP

    r1 = np.sqrt(x1**2 + y1**2) # distance of 1st body to sun
    r2 = np.sqrt(x2**2 + y2**2) # distance of 2nd body to sun
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)    # distance between 1st and 2nd body

    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2-x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2-y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1-x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1-y2) / d**3)

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

def plot_two_body_orbit(solution):

    two_body_solution = solution

    two_body_x1 = two_body_solution[:, 0]
    two_body_y1 = two_body_solution[:, 1]
    two_body_x2 = two_body_solution[:, 4]
    two_body_y2 = two_body_solution[:, 5]

    plt.plot(two_body_x1,two_body_y1)
    plt.plot(two_body_x2,two_body_y2)
    plt.show()

def plot_fourier(solutions):

    for solution in solutions:
        values = solution[:, 0]
        fourier = np.fft.rfft(values)
        plt.plot(freq,np.abs(fourier))
    plt.show()

def plot_fourier_difference(solution_1, solution_2):

    solution_1_values = solution_1[:, 0]
    solution_2_values = solution_2[:, 0]

    solution_1_fourier = np.fft.rfft(solution_1_values)
    solution_2_fourier = np.fft.rfft(solution_2_values)
    
    max_frequency = int(0.02 * time_steps / samples_per_time)
    truncated_range = freq[:max_frequency]

    plt.plot(truncated_range,np.abs(solution_1_fourier)[:max_frequency], color="blue", label="theoretical orbit")
    plt.plot(truncated_range,np.abs(solution_2_fourier)[:max_frequency], color="orange", label="actual orbit")
    # plt.plot(truncated_range,np.abs(solution_1_fourier - solution_2_fourier), color="red")
    plt.plot(truncated_range,(solution_1_fourier - solution_2_fourier)[:max_frequency], color="red", label="real difference")
    plt.plot(truncated_range,(solution_1_fourier - solution_2_fourier)[:max_frequency] * -1j, color="black", label="imaginary difference")
    plt.title("Difference in theoretical and observed")
    plt.legend()
    # plt.show()

examples = [
    # [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01), 100000, 1, 1],   # coil orbit
    # [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1], # messier coild orbit
    # [1, 0, 0, np.sqrt(G*10000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01), 100000, 1, 100], # massive 2nd body
    [1, 0, 0.001, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1], # two bodies, minor x velocity 
    [1, 0, 0.001, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 2], 
    [1, 0, 0.001, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 5],
    # [1, 0, 0, np.sqrt(G*100000/1), 1.5, 0, 0, np.sqrt(G*100000/2), 100000, 1000, 1], # two bodies, massive 1st body
    # [1, 0, 0, np.sqrt(G*100000/1), 1.1, 0, 0, np.sqrt(G*100000/1.1), 100000, 100, 1], # two bodies, (less) massive 1st body 
    # [1, 0, 0, np.sqrt(G*100000/1), -1.01, 0, 0, np.sqrt(G*100000/1.01), 10000, 1, 1], # two bodies travelling opposite directions
]

fig = plt.figure(figsize=(8,6))

for i in range(len(examples)): 
    plt.subplot(len(examples), 1, i+1)
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples[i]
    two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
    one_body_solution = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))
    # plot_two_body_orbit(two_body_solution)
    plot_fourier_difference(one_body_solution, two_body_solution) 
plt.tight_layout()
plt.show()
