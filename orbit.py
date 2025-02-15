# Jason French 1.15.24

import numpy as np
from scipy.integrate import odeint
import scipy as sci
import matplotlib.pyplot as plt
import samples


# Constatnts
G = samples.G

# Simulation setup
samples_per_time = 128
time_steps = 100000 * 1
t = np.linspace(0, time_steps, time_steps*samples_per_time) 
freq = np.fft.rfftfreq(len(t), 1/samples_per_time) 

# signal = np.sin(2*np.pi*t)
# S = np.fft.rfft(signal)
# plt.plot(freq, S)
# plt.show()

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

def plot_fourier_difference(solution_1, solution_2, title):

    solution_1_values = solution_1[:, 0]
    solution_2_values = solution_2[:, 0]

    solution_1_fourier = np.fft.rfft(solution_1_values)
    solution_2_fourier = np.fft.rfft(solution_2_values)

    min_f = 0.0003
    max_f = 0.0005
    min_frequency = int(min_f * time_steps)
    max_frequency = int(max_f * time_steps)
    truncated_range = freq[min_frequency:max_frequency]

    plt.plot(truncated_range,np.abs(solution_1_fourier)[min_frequency:max_frequency], color="blue", label="theoretical orbit")
    plt.plot(truncated_range,np.abs(solution_2_fourier)[min_frequency:max_frequency], color="orange", label="actual orbit")
    plt.plot(truncated_range,(solution_1_fourier - solution_2_fourier)[min_frequency:max_frequency], color="red", label="real difference")
    plt.plot(truncated_range,(solution_1_fourier - solution_2_fourier)[min_frequency:max_frequency] * -1j, color="black", label="imaginary difference")
    plt.title(title)
    plt.legend(loc="upper right")
    # plt.show()

labels = [
    "x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "M", "m1", "m2"
]

# examples = samples.increasing_vx
# examples = samples.decreasing_dist
# examples = samples.increasing_both_mass
# examples = samples.increasing_both_masses_and_distances
# examples = samples.decreasing_dist_opposing_orbits
# examples = samples.eliptical

# examples = samples.decreasing_v2y
# examples = samples.decreasing_dist
examples = samples.increasing_2nd_body_mass


categories=[]
heights=[]

fig = plt.figure(figsize=(8,6))

for i in range(len(examples)): 
    plt.subplot(len(examples), 1, i+1)
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = examples[i]
    two_body_solution = odeint(two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
    one_body_solution = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))
    # plot_two_body_orbit(two_body_solution)
    if i == 0:
        title = "Base orbit"
    else:
        title = ""
        for j in range(len(examples[i])):
            title += "" if examples[i][j] == examples[0][j] else f"{labels[j]}={examples[i][j]} "     # add parameter to title if it doesnt match base state
    plot_fourier_difference(one_body_solution, two_body_solution, title)


    # Plot variation in peaks
    categories.append(title)
    x_values = two_body_solution[:, 0]  # Extract x coordinate
    # x_values_one = one_body_solution[:, 0]  # Extract x coordinate
    x_hat = np.fft.rfft(x_values)       # Compute FFT
    # x_hat_one = np.fft.rfft(x_values_one)       # Compute FFT
    F_upper = 10000 
    height = 200 
    peaks, _ = sci.signal.find_peaks(np.abs(x_hat[:F_upper]), height=height)  # Find peaks
    # peaks_one, _ = sci.signal.find_peaks(np.abs(x_hat_one[:F_upper]), height=height)  # Find peaks
    plt.scatter(freq[peaks], np.abs(x_hat[peaks]), c='orange')

    tallest_peak_index = peaks[np.argmax(np.abs(x_hat[peaks]))]  # Index of max peak
    tallest_peak_freq = freq[tallest_peak_index]  # Frequency of max peak
    plt.scatter(tallest_peak_freq, np.abs(x_hat[tallest_peak_index]), c='red', label="Tallest Peak", marker="x", s=100)

    # tallest_peak_index_one = peaks[np.argmax(np.abs(x_hat_one[peaks]))]  # Index of max peak
    # tallest_peak_freq_one = freq[tallest_peak_index]  # Frequency of max peak
    # plt.scatter(tallest_peak_freq_one, np.abs(x_hat_one[tallest_peak_index_one]), c='red', label="Tallest Peak", marker="x", s=100)

    # heights.append(np.abs(x_hat[tallest_peak_index])- np.abs(x_hat_one[tallest_peak_index_one]))

    print("difference in indices", tallest_peak_index)

plt.tight_layout()
plt.show()



# Create the bar chart
plt.bar(categories, heights, color='orange')
print(heights)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Heights')
plt.title('Height of Max Peak')

# Show the plot
plt.show()
