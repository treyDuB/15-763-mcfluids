import numpy as np
import os

def generate(side_length, dx):
    # sample nodes uniformly on a square
    n_seg = int(side_length / dx)
    x = np.array([[0.0, 0.0]] * ((n_seg + 1) ** 2))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]

    return [x]

def generate_3D(side_length, dx):
    # sample nodes uniformly on a cube
    n_seg = int(side_length / dx)
    x = np.array([[0.0, 0.0, 0.0]] * ((n_seg + 1) ** 3))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            for k in range(0, n_seg + 1):
                x[i * (n_seg + 1) ** 2 + j * (n_seg + 1) + k] = [-side_length / 2 + i * step,
                                                                   -side_length / 2 + j * step,
                                                                   -side_length / 2 + k * step]

    return [x]

def perturb(x, dx):
    # perturb the nodes uniformly in a cube of size dx
    for i in range(len(x)):
        x[i] += np.random.uniform(-dx / 2, dx / 2,)

    return x


def write_to_file(frameNum, x, num_particles):
    # Check if 'output' directory exists; if not, create it
    if not os.path.exists('output'):
        os.makedirs('output')

    # create obj file
    filename = f"output/{frameNum}.obj"
    with open(filename, 'w') as f:
        # write particle coordinates
        for i in range(num_particles):
            row = x[i]
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n") 
