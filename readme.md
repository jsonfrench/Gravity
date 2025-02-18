<a name ="readme-top"></a>

<h2 align="center">Identifying other Planets Based on Variations in Earth's Orbit</h2>

Knowing only information about the orbit of your home planet, could you tell where to look in the sky to find other planets in your solar system? This repository contains simulations and computations wheer we attempt to answer this question using various techniques like the Fourier transform and wavelets. 

<h2 align="center">Simulating a Two Body System</h2>

We use the ODEint library to simulate our system. 

```python
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
```

The above code produces a decent simulation of two planets orbiting a massive central body. Below is a depiction of one of these simulated orbits.

![two_body_orbit](https://github.com/jsonfrench/Gravity/blob/main/media/example_orbit.png)

<h2 align="center">Fourier Transform</h2>

One method for analytically studying these orbits is using the fourier transform. Here is what the fourier transform of the previously depicted orbit looks like:

![two_body_fourier_transform](https://github.com/jsonfrench/Gravity/blob/main/media/example_fourier.png)
