import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import samples

def one_body(IVP, t, G, M): 
    
    x, y, vx, vy = IVP  # unpack initial value problem 

    r = np.sqrt(x**2 + y**2)    # distance from sun to body

    ax = -G * M * x / r**3  # calculute force of sun on the body
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

def two_body(IVP, t, G, M, m1, m2):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP

    r1 = np.sqrt(x1**2 + y1**2)  # distance of 1st body to sun
    r2 = np.sqrt(x2**2 + y2**2)  # distance of 2nd body to sun
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # distance between 1st and 2nd body

    ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2 - x1) / d**3)
    ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2 - y1) / d**3)
    ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1 - x2) / d**3)
    ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1 - y2) / d**3)

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

def orbital_error(theoretical, actual, index_observation: int) -> tuple:
    theor_x = theoretical[:,0]
    theor_y = theoretical[:,1]
    actual_x = actual[:,0]
    actual_y = actual[:,1]

    x_error = np.abs(theor_x[index_observation] - actual_x[index_observation])
    y_error = np.abs(theor_y[index_observation] - actual_y[index_observation])

    L2_error = np.sqrt(x_error**2 + y_error**2)

    return (x_error,y_error,L2_error)

def two_body_distance(actual, index_observation: int) -> tuple:
    actual_x1 = actual[:,0]
    actual_y1 = actual[:,1]
    actual_x2 = actual[:,4]
    actual_y2 = actual[:,5]

    x_dist = np.abs(actual_x2[index_observation] - actual_x1[index_observation])
    y_dist = np.abs(actual_y2[index_observation] - actual_y1[index_observation])

    L2_dist = np.sqrt(x_dist**2 + y_dist**2)

    return (L2_dist)

# Define Constants
G = 6.67 * 10 ** (-11)
samples_per_time = 128
time_steps = 50000
t = np.linspace(0, time_steps, time_steps*samples_per_time)
freq = np.fft.rfftfreq(len(t), 1/samples_per_time)  # frequencies to be used by the discrete fourier transform 

# IVP = [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1]
IVP = samples.decreasing_dist[6]
# IVP = samples.decreasing_dist[7]
# IVP = samples.decreasing_dist[8]

x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = IVP
actual_solution = odeint(two_body, IVP[0:8], t, args=(G, IVP[8], IVP[9], IVP[10]))
actual_solution_thetas = np.arctan2(actual_solution[:,1], actual_solution[:,0])
theoretical_solution = odeint(one_body, [x1, y1, vx1, vy1], t, args=(G, m0))
theoretical_solution_thetas = np.arctan2(theoretical_solution[:,1], theoretical_solution[:,0])
index_max = time_steps #* samples_per_time
min_frequency = 0 #40
max_frequency = 100 

# ax = plt.figure().add_subplot(projection='3d')

# z_resolution = 100
# increment = time_steps * samples_per_time // z_resolution
# for i in range(z_resolution):
#     print(f"plotting {i+1}/{z_resolution}")
#     x = freq[min_frequency:max_frequency]
#     y =  np.abs(np.fft.rfft(actual_solution[:increment * (i+1)]))[min_frequency:max_frequency]
#     ax.plot(x, y, zs=i, zdir='x')

zeros = []  # list of time points when new year occurs
thetas = actual_solution_thetas[:time_steps*samples_per_time]
for i in range ((len(thetas))-1):
    if thetas[i] - thetas[i+1] > 0:
        zeros.append((i+0.5) / samples_per_time)

# for i in range(len(zeros)):
#     print(f"year {i+1}: {zeros[i]}")

year_length = np.array([])
year_error = np.array([])
for i in range(len(zeros)-1):
    # print(f"year {i+2} length: {zeros[i+1] - zeros[i]}")
    year_length = np.append(year_length, zeros[i+1] - zeros[i]) # add year lengths to list
    year_error = np.append(year_error, two_body_distance(actual_solution,int(zeros[i])))

avg_dist_per_year = np.array([])
for i in range(len(zeros)-1):
    sum = 0 
    for j in range(int(zeros[i]), int(zeros[i+1])):
        sum += two_body_distance(actual_solution, int(j))
    avg = sum / (zeros[i+1] - zeros[i])
    avg_dist_per_year = np.append(avg_dist_per_year, avg)

print(year_length.size)
print(year_error.size)

print("[", end="")
for length in year_length:
    print(length,",", end="")
print("]")


tx = plt.figure().add_subplot()
tx.plot(np.linspace(2,len(zeros), len(zeros)-1), year_length)
tx.set_title("Body 1 Year length")
tx.set_ylabel("length of year")
tx.set_xlabel("year")
plt.show()

tx = plt.figure().add_subplot()
tx.plot(np.linspace(2,len(avg_dist_per_year), len(avg_dist_per_year)), avg_dist_per_year, color="lime")
tx.set_title("Average L2 Distance per year")
tx.set_ylabel("distance")
tx.set_xlabel("year")
plt.show()


year_length_fft = np.abs(np.fft.rfft(year_length))

# for x in range(len(spectrums)):
    #plot spectrums

# Make legend, set axes limits and labels
# ax.legend()
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_xlabel('Time')
# ax.set_ylabel('Frequency')
# ax.set_zlabel('Amplitude')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
# ax.view_init(elev=20., azim=-35, roll=0)

# plt.show()

