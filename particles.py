import numpy as np
import os

def generate(side_length, dx, random=False):
    # add points points uniformly on a square
    n_seg = int(side_length / dx)
    x = np.array([[0.0, 0.0]] * ((n_seg + 1) ** 2))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            rand_x = np.random.uniform(-dx / 4, dx / 4) if random else 0
            rand_y = np.random.uniform(-dx / 4, dx / 4) if random else 0

            p = [-side_length / 2 + i * step + rand_x, -side_length / 2 + j * step + rand_y]
            x[i * (n_seg + 1) + j] = p

    return [x]


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
