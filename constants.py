import numpy as np

# Constants
G = 6.674e-11
labels = ["x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "M", "m1", "m2"]

# Simulation parameters
samples_per_time = 128
time_steps = 100 * 1
t = np.linspace(0, time_steps, time_steps*samples_per_time) # array of time points to solve the system at
freq = np.fft.rfftfreq(len(t), 1/samples_per_time)  # frequencies to be used by the discrete fourier transform 

