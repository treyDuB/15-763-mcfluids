import numpy as np

def kineticEnergy(m, velocities):
    sum = 0.0
    for i in range(len(velocities)):
        sum += 0.5 * m * np.linalg.norm(velocities[i]) ** 2
    return sum