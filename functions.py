import numpy as np  # numpy for linear algebra


def G_2D(x, y):
    # 2D Green's function
    r = np.sqrt(x*x + y*y)
    return (0.5 / np.pi) * np.log(r)

def G_3D(x, y, z):
    # 3D Green's function
    r = np.sqrt(x*x + y*y + z*z)
    return (1.0 / (4.0 * np.pi)) * (1.0 / r)

# returns 2x? vector??
def grad_G_2D(x, y):
    # Gradient of 2D Green's function
    r = np.sqrt(x*x + y*y)
    return np.array([-y / (np.pi * r**2), x / (np.pi * r**2)])

def grad_G_x_2D(x,y):
    # Gradient of 2D Green's function w.r.t. x
    r = np.linalg.norm(y-x)
    return (x-y)/(r**3)/4./np.pi

def grad_G_3D(x, y, z):
    # Gradient of 3D Green's function
    r = np.sqrt(x*x + y*y + z*z)
    return np.array([-x / (4.0 * np.pi * r**3), -y / (4.0 * np.pi * r**3), -z / (4.0 * np.pi * r**3)])

def Hessian_G_2D(x, y):
    # Hessian of 2D Green's function
    r = np.sqrt(x*x + y*y)
    return np.array([[x**2 / (np.pi * r**4), y**2 / (np.pi * r**4)],
                     [y**2 / (np.pi * r**4), x**2 / (np.pi * r**4)]]) - \
           np.array([[1 / (np.pi * r**2), 0],
                     [0, 1 / (np.pi * r**2)]])

# returns 2x2 matrix
def S_2D(x, y):
    # 2D S function
    r = y - x
    unit_SA = 2 * np.pi
    r_norm = np.linalg.norm(r)
    return (1 / (unit_SA * r_norm**4)) * (2 * np.outer(r, r) - np.eye(2) * r_norm**2)

def S_3D(x, y):
    # 3D S function
    r = y - x
    unit_SA = 4 * np.pi
    r_norm = np.linalg.norm(r)
    return (1 / (unit_SA * r_norm**5)) * (3 * np.outer(r, r) - np.eye(3) * r_norm**2)

