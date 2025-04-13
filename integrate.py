# Integration helpers

import numpy as np  # numpy for linear algebra

def cubicInterpolate(p, x):
    return p[1] + 0.5 * x*(p[2] - p[0] + 
                           x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + 
                              x*(3.0*(p[1] - p[2]) + p[3] - p[0])))
                        


def bicubicInterpolate(p, x, y):
    arr = [0] * 4
    arr[0] = cubicInterpolate(p[0], x)
    arr[1] = cubicInterpolate(p[1], x)
    arr[2] = cubicInterpolate(p[2], x)
    arr[3] = cubicInterpolate(p[3], x)
    return cubicInterpolate(arr, y)

def tricubicInterpolate(p, x, y, z):
    arr = [0] * 4
    arr[0] = bicubicInterpolate(p[0], x, y)
    arr[1] = bicubicInterpolate(p[1], x, y)
    arr[2] = bicubicInterpolate(p[2], x, y)
    arr[3] = bicubicInterpolate(p[3], x, y)
    return cubicInterpolate(arr, z)

def linearInterpolate(p, x):
    return p[0] + x * (p[1] - p[0])

def bilinearInterpolate(p, x, y):
    arr = [0] * 2
    arr[0] = linearInterpolate(p[0], x)
    arr[1] = linearInterpolate(p[1], x)
    return linearInterpolate(arr, y)

def trilinearInterpolate(p, x, y, z):
    arr = [0] * 2
    arr[0] = bilinearInterpolate(p[0], x, y)
    arr[1] = bilinearInterpolate(p[1], x, y)
    return linearInterpolate(arr, z)

def evaluateComponentVelocityLinear(position, gridOffSet, grid, dx, height):
    # Calculate the grid cell indices
    position -= gridOffSet
    i = int(position[0] / dx)
    j = int(position[1] / dx)
    # k = int(position[2] / grid[2])

    # Calculate the local coordinates within the cell
    # x = (position[0] - (gridOffSet[0] + i * grid[0])) / grid[0]
    # y = (position[1] - (gridOffSet[1] + j * grid[1])) / grid[1]
    # z = (position[2] - (gridOffSet[2] + k * grid[2])) / grid[2]

    # Get the 2x2 surrounding points
    p = np.zeros((2, 2))
    for di in range(0, 2):
        for dj in range(0, 2):
                p[di][dj] = grid[(i + di) * height + (j + dj)]

    gpos = np.array([i, j]) * dx
    interp = (position - gpos) / dx

    # Interpolate the velocity using bilinear interpolation
    return bilinearInterpolate(p, interp[0], interp[1])


def evaluateComponentVelocityCubic(position, gridOffSet, grid, dx):
    # Calculate the grid cell indices
    position -= gridOffSet
    i = int(position[0] / dx)
    j = int(position[1] / dx)
    k = int(position[2] / dx)

    # # Calculate the local coordinates within the cell
    # x = (position[0] - (gridOffSet[0] + i * grid[0])) / grid[0]
    # y = (position[1] - (gridOffSet[1] + j * grid[1])) / grid[1]
    # z = (position[2] - (gridOffSet[2] + k * grid[2])) / grid[2]

    # # Get the 4x4x4 surrounding points
    # p = np.zeros((4, 4, 4))
    # for di in range(0, 4):
    #     for dj in range(0, 4):
    #         for dk in range(0, 4):
    #             p[di][dj][dk] = getGridPoint(i + di - 1, j + dj - 1, k + dk - 1)

    # gpos = np.array([i, j, k]) * dx
    # interp = (position - gpos) / dx

    # Get the 4x4 surrounding points
    p = np.zeros((4, 4))
    for di in range(0, 4):
        for dj in range(0, 4):
            p[di][dj] = grid[i + di - 1][j + dj - 1]
    
    gpos = np.array([i, j]) * dx
    interp = (position - gpos) / dx

   
    # Interpolate the velocity using bicubic interpolation
    # return tricubicInterpolate(p, interp[0], interp[1], interp[2])
    return bicubicInterpolate(p, interp[0], interp[1])


def evaluateVelocityAtPosition(position, dx, gridU, gridV, height):  # gridW=None):
    hdx = 0.5 * dx
    offsetU = np.array([0.0, hdx])
    offsetV = np.array([hdx, 0.0])
    # offsetW = np.array([hdx, hdx, 0.0])

    vx = evaluateComponentVelocityLinear(position, offsetU, gridU, dx, height)
    vy = evaluateComponentVelocityLinear(position, offsetV, gridV, dx, height)
    # vz = evaluateComponentVelocityLinear(position, offsetW, gridW, dx)

    return np.array([vx, vy])  # , vz])

