# Jason French 1.15.24

import numpy as np
from scipy.integrate import odeint
import scipy as sci
import constants

class Orbit:

    initial_state = []
    G = 0
    samples_per_time = 0
    time_steps = 0 
    t = np.array([]) 
    freq = np.array([]) 
    one_body_solution = np.array([])
    two_body_solution = np.array([])

    def __init__(self, initial_state, samples_per_time=constants.samples_per_time, time_steps=constants.time_steps, t=constants.t, freq=constants.freq, G=constants.G):

        self.samples_per_time = samples_per_time
        self.initial_state = initial_state
        self.time_steps = time_steps 
        self.freq = freq 
        self.G = G
        self.t = t 

        x1, y1, vx1, vy1, x2, y2, vx2, vy2, m0, m1, m2 = initial_state
        self.one_body_solution = odeint(self.one_body, [x1, y1, vx1, vy1], t, args=(G, m0, m1))
        self.two_body_solution = odeint(self.two_body, [x1, y1, vx1, vy1, x2, y2, vx2, vy2], t, args=(G, m0, m1, m2))
    
    def one_body(IVP, t, G, M): 
        
        x, y, vx, vy = IVP  

        r = np.sqrt(x**2 + y**2)    # distance between the body and the sun

        ax = -G * M * x / r**3  
        ay = -G * M * y / r**3

        return [vx, vy, ax, ay] 

    def two_body(IVP, t, G, M, m1, m2): 
        
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = IVP    

        r1 = np.sqrt(x1**2 + y1**2)     # distance of 1st body to sun
        r2 = np.sqrt(x2**2 + y2**2)     # distance of 2nd body to sun
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2)    # distance between 1st and 2nd body

        ax1 = (-G * M * x1 / r1**3) + (-G * m2 * (x2-x1) / d**3)   
        ay1 = (-G * M * y1 / r1**3) + (-G * m2 * (y2-y1) / d**3)
        ax2 = (-G * M * x2 / r2**3) + (-G * m1 * (x1-x2) / d**3)
        ay2 = (-G * M * y2 / r2**3) + (-G * m1 * (y1-y2) / d**3)

        return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

    def compute_fourier(solution):
        return np.fft.rfft(solution)
