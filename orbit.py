# Jason French 1.15.24

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Constatnts
G = 6.674e-11

# Parameters
m1 = 100000     # Large mass (sun)
m2 = 1          # small mass (planet)

# Initial Values
x1 = 0         
y1 = 0         
x2 = 1         
y2 = 0         
vx1 = 0        
vy1 = 0        
vx2 = 0        
vy2 = np.sqrt(G*m1) * 1.3

def dSdt(IVP, t, G, M): 
    
    x, y, vx, vy = IVP

    r = np.sqrt(x**2 + y**2)

    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return [vx, vy, ax, ay]

t = np.linspace(0, 10000, 100000)

solution = odeint(dSdt, [x2, y2, vx2, vy2], t, args=(G, m1))

x=solution[:,0]
y=solution[:,1]

plt.plot(x,y)
plt.axis("equal")
plt.show()