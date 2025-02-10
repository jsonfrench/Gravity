import numpy as np

G = 6.674e-11

# "x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2", "M", "m1", "m2"


increasing_2nd_body_mass = [
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 10],
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 100],
    ]
increasing_vx = [
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0.01, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0.1, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    ]
decreasing_v2y = [
    [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1], 
    [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.98, 100000, 1, 1], 
    [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.97, 100000, 1, 1], 
]
decreasing_dist = [
    [1, 0, 0, np.sqrt(G*100000/1), 1.3, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0, np.sqrt(G*100000/1), 1.2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0, np.sqrt(G*100000/1), 1.1, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
]
increasing_both_mass = [
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 10, 10],
    [1, 0, 0, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 100, 100],
]
decreasing_dist_opposing_orbits = [
    [1, 0, 0, np.sqrt(G*100000/1), -1.3, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0, np.sqrt(G*100000/1), -1.2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
    [1, 0, 0, np.sqrt(G*100000/1), -1.1, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1],
]


# Old Examples:
    # [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01), 100000, 1, 1],   # coil orbit
    # [1, 0, 0, np.sqrt(G*100000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01) * 0.99, 100000, 1, 1], # messier coild orbit
    # [1, 0, 0, np.sqrt(G*10000/1), 1.01, 0, 0, np.sqrt(G*100000/1.01), 100000, 1, 100], # massive 2nd body
    # [1, 0, 0.001, np.sqrt(G*100000/1), 2, 0, 0, np.sqrt(G*100000/2), 100000, 1, 1], # two bodies, minor x velocity 
    # [1, 0, 0, np.sqrt(G*100000/1), 1.5, 0, 0, np.sqrt(G*100000/2), 100000, 1000, 1], # two bodies, massive 1st body
    # [1, 0, 0, np.sqrt(G*100000/1), 1.1, 0, 0, np.sqrt(G*100000/1.1), 100000, 100, 1], # two bodies, (less) massive 1st body 
    # [1, 0, 0, np.sqrt(G*100000/1), -1.01, 0, 0, np.sqrt(G*100000/1.01), 10000, 1, 1], # two bodies travelling opposite directions

