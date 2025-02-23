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

def plot_peak_width(solution, title):

    solution_values = solution[:, 0]

    solution_fourier = np.fft.rfft(solution_values)


    x_values = two_body_solution[:, 0]  # Extract x coordinate
    x_values_one = one_body_solution[:, 0]  # Extract x coordinate
    x_hat = np.fft.rfft(x_values)       # Compute FFT
    x_hat_one = np.fft.rfft(x_values_one)       # Compute FFT
    F_upper = 10000 
    height_threshold = 2000

    peaks, _ = sci.signal.find_peaks(np.abs(x_hat), height=height_threshold)  
    tallest_peak_index = peaks[np.argmax(np.abs(x_hat[peaks]))]  # Index of max peak
    width = sci.signal.peak_widths(np.abs(x_hat), [tallest_peak_index])[0][0] / samples_per_time
    print("width:",width)
    print("tallest peak index (plot peak width): ", tallest_peak_index)

    tallest_peak_freq = freq[tallest_peak_index]  # Frequency of max peak

    results_half = sci.signal.peak_widths(solution_values, peaks, rel_height=0.5)
    results_half[0]  # widths

    plt.plot(solution_values)
    plt.plot(peaks, solution_values[peaks], "solution_values")
    plt.hlines(*results_half[1:], color="C2")
    # plt.show()


    min_f = tallest_peak_freq - (width / (2))   
    max_f = tallest_peak_freq + (width / (2))
    min_frequency = int(min_f * time_steps)
    max_frequency = int(max_f * time_steps)
    truncated_range = freq[min_frequency:max_frequency]
    # plt.plot(truncated_range,np.abs(solution_fourier)[min_frequency:max_frequency], color="green", label="theoretical orbit")

    # plt.title(title)
    # plt.legend(loc="upper right")
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

examples = []
num=50
for i in range(num):
    examples.append(
        [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * (1.0 + i*(0.4/(num+1))), 100000, 1, 1], 
    )

# examples = samples.decreasing_dist
# examples = samples.increasing_2nd_body_mass

categories=[]
peak_heights=[]
peak_widths=[]

masses=np.array([])
distances = np.array([])
eccentricities = np.array([])

fig = plt.figure(figsize=(8,6))

for i in range(len(examples)): 

    print(f"simulating: {i+1}/{len(examples)}")

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
    # plot_peak_width(two_body_solution, "Isolated spike")


    # Plot variation in peaks
    categories.append(title)
    x_values = two_body_solution[:, 0]  # Extract x coordinate
    x_values_one = one_body_solution[:, 0]  # Extract x coordinate
    x_hat = np.fft.rfft(x_values)       # Compute FFT
    x_hat_one = np.fft.rfft(x_values_one)       # Compute FFT
    F_upper = 10000 
    height_threshold = 2000
    peaks, _ = sci.signal.find_peaks(np.abs(x_hat[:F_upper]), height=height_threshold)  # Find peaks
    peaks_one, _ = sci.signal.find_peaks(np.abs(x_hat_one[:F_upper]), height=height_threshold)  # Find peaks
    plt.scatter(freq[peaks], np.abs(x_hat[peaks]), c='orange')

    print("peaks: ", peaks)
    widths = sci.signal.peak_widths(np.abs(x_hat[:F_upper]), peaks)
    print("peak widths:", widths)

    tallest_peak_index = peaks[np.argmax(np.abs(x_hat[peaks]))]  # Index of max peak
    tallest_peak_freq = freq[tallest_peak_index]  # Frequency of max peak
    plt.scatter(tallest_peak_freq, np.abs(x_hat[tallest_peak_index]), c='orange', label="Tallest Peak", marker="x", s=100)
    print("tallest peak index (two body orbit): ", tallest_peak_index)


    tallest_peak_index_one = peaks[np.argmax(np.abs(x_hat_one[peaks]))]  # Index of max peak for one body orbit
    tallest_peak_freq_one = freq[tallest_peak_index]  # Frequency of max peak for one body orbit
    plt.scatter(tallest_peak_freq_one, np.abs(x_hat_one[tallest_peak_index_one]), c='blue', label="Tallest Peak", marker="x", s=100)



    peak_heights.append(np.abs(x_hat[tallest_peak_index])-np.abs(x_hat_one[tallest_peak_index_one]))
    # peak_widths.append(sci.signal.peak_widths(np.abs(x_hat[:F_upper]), [tallest_peak_index]))

    # print("difference in indices", tallest_peak_index - tallest_peak_index_one)

    eccentricities = np.append(eccentricities,[examples[i][7] / np.sqrt(G*100000/1.01)])
    distances = np.append(distances,examples[i][4]-examples[i][0])
    masses = np.append(masses,examples[i][10])

plt.tight_layout()
plt.show()

print("peak heights:", peak_heights)
print("peak widths:", peak_widths)

for i in range(len(peak_heights)):
    # plt.scatter(masses[i],peak_heights[i])
    # print(f"({masses[i]},{peak_heights[i]})")
    # plt.scatter(masses[i],peak_widths[i])
    # print(f"({masses[i]},{peak_widths[i]})")

    # plt.scatter(distances[i],peak_heights[i], c="orange")
    # print(f"({distances[i]},{peak_heights[i]})")

    plt.scatter(eccentricities[i],peak_heights[i],c="orange")

plt.plot(masses,1791.87136*masses + 951.85075)
# plt.plot(distances,2468.88*distances**-1.76386)

plt.show()



# Create the bar chart
plt.bar(categories, peak_heights, color='orange')
print(peak_heights)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('peak_heights')
plt.title('Height of Max Peak')

# Show the plot
plt.show()
