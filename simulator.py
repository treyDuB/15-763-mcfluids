# PIC/FLIP Fluids

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
pygame.init()

import particles   # particles set up
import energy
# import time_integrator


U_FIELD = 0
V_FIELD = 1

AIR_CELL = 0
FLUID_CELL = 1
SOLID_CELL = 2

def clamp(x, a, b):
    return min(max(x, a), b)

# ANCHOR: sim_setup
# simulation setup
side_len = 0.45
E = 1e5         # Young's modulus
nu = 0.4        # Poisson's ratio
n_seg = 2       # num of segments per side of the square

# DBC = [(n_seg + 1) * (n_seg + 1) * 2]   # dirichlet node index
# DBC_v = [np.array([0.0, -0.5])]         # dirichlet node velocity
# DBC_limit = [np.array([0.0, -0.7])]     # dirichlet node limit position
ground_n = np.array([0.0, 1.0])         # normal of the slope
ground_n /= np.linalg.norm(ground_n)    # normalize ground normal vector just in case
ground_o = np.array([0.0, 75.0])        # a point on the slope  

left_n = np.array([1.0, 0.0])           # normal of the left wall
left_n /= np.linalg.norm(left_n)        # normalize left wall normal vector just in case
left_o = np.array([50.0, 0.0])         # a point on the left wall

right_n = np.array([-1.0, 0.0])         # normal of the right wall
right_n /= np.linalg.norm(right_n)      # normalize right wall normal vector just in case
right_o = np.array([150.0, 0.0])         # a point on the right wall


mu = 0.4        # friction coefficient of the slope

print("ground_n: ", ground_n)
print("ground_o: ", ground_o) 



#fluid setup

gravity = [0.0, -9.8]


width = 200
height = 200

f_spacing = 5
f_num_X = int(width / f_spacing) + 1  # number of grid cell walls?
f_num_Y = int(height / f_spacing) + 1 # ^
h = max([float(width / f_num_X), float(height / f_num_Y)])
f_inv_spacing = 1.0 / h
f_num_cells = f_num_X * f_num_Y

print(f_num_X, f_num_Y, h, f_num_cells)

dt = 0.025


# grid setup
u = [0.0] * f_num_cells
v = [0.0] * f_num_cells
du = [0.0] * f_num_cells
dv = [0.0] * f_num_cells
prev_U = [0.0] * f_num_cells
prev_V = [0.0] * f_num_cells
p = [0.0] * f_num_cells              # what does this mean?
s = [0.0] * f_num_cells              # this too
cell_type = [AIR_CELL] * f_num_cells
cell_color = [0.0] * f_num_cells

# particle setup
max_particles = 1000
particle_pos = np.array([[0.0, 0.0]] * max_particles)
particle_vel = np.array([[0.0, 0.0]] * max_particles)


# particle_type = [0] * num_particles
# particle_color = [0.0, 0.0, 0.0] * num_particles
particle_density = [0.0] * f_num_cells
particle_rest_density = 0.0


particle_radius = 0.4 * h
particle_mass = 0.8 * particle_radius * particle_radius * particle_radius * rho
p_spacing = 2.0
dx = p_spacing * particle_radius
p_inv_spacing = 1.0 / dx
p_num_X = int(width * p_inv_spacing) + 1
p_num_Y = int(height * p_inv_spacing) + 1
p_num_cells = p_num_X * p_num_Y

num_cell_particles = [0] * p_num_cells
first_cell_particle = [0] * (p_num_cells+1) # why +1?
cell_particle = [0] * max_particles
max_particles_per_cell = 10

[xtmp] = particles.generate(50, dx, True)
num_particles = len(xtmp)
for i in range(num_particles):
    if(i < max_particles):
        particle_pos[i] = (xtmp[i]) + [width/2 - 25, height / 2]
        particle_vel[i] = [0.0, -5.0]
    else:
        break


def integrateParticles(dt : float, gravity : list[float]):
    global particle_pos, particle_vel, num_particles, width, height, ground_o, ground_n

    for i in range(0, num_particles):
        particle_vel[i][0] += gravity[0] * dt
        particle_vel[i][1] += gravity[1] * dt

        particle_pos[i][0] += particle_vel[i][0] * dt
        particle_pos[i][1] += particle_vel[i][1] * dt

        x = particle_pos[i][0]
        y = particle_pos[i][1]

        if x < left_o[0]:
            particle_pos[i][0] = left_o[0]
            particle_vel[i][0] = 0.0
        if x >= right_o[0]:
            particle_pos[i][0] = right_o[0]
            particle_vel[i][0] = 0.0
        if y < ground_o[1]:
            particle_pos[i][1] = ground_o[1]
            particle_vel[i][1] = 0.0
        if y >= height:
            particle_pos[i][1] = height
            particle_vel[i][1] = 0.0

def pushParticlesApart(num_iterations : int):
    global particle_pos, num_particles, p_num_X, p_num_Y, p_inv_spacing, p_num_cells, particle_radius
    global num_cell_particles, first_cell_particle, cell_particle

    # Count per cell
    num_cell_particles = [0] * p_num_cells
    for i in range(num_particles):
        x = particle_pos[i]
        xi = clamp(int(x[0] * p_inv_spacing), 0, p_num_X - 1)
        yi = clamp(int(x[1] * p_inv_spacing), 0, p_num_Y - 1)
        num_cell_particles[xi * p_num_Y + yi] += 1
    
    # partial sum

    first = 0

    for i in range(p_num_cells):
        first += num_cell_particles[i]
        first_cell_particle[i] = first
    
    first_cell_particle[p_num_cells] = first # total number of particles

    # sort particles
    for i in range(num_particles):
        x = particle_pos[i]
        xi = clamp(int(x[0] * p_inv_spacing), 0, p_num_X - 1)
        yi = clamp(int(x[1] * p_inv_spacing), 0, p_num_Y - 1)
        cell = xi * p_num_Y + yi
        index = first_cell_particle[cell] - 1
        cell_particle[index] = i
        first_cell_particle[cell] -= 1

    # push particles apart
    min_dist = 2.0 * particle_radius
    min_dist2 = min_dist * min_dist

    for iter in range(num_iterations):
        for i in range(num_particles):
            p1 = particle_pos[i]
            pxi = clamp(int(p1[0] * p_inv_spacing), 0, p_num_X - 1)
            pyi = clamp(int(p1[1] * p_inv_spacing), 0, p_num_Y - 1)
            x0 = max(0, pxi - 1)
            x1 = min(p_num_X - 1, pxi + 1)
            y0 = max(0, pyi - 1)
            y1 = min(p_num_Y - 1, pyi + 1)

            for xi in range(x0, x1 + 1):
                for yi in range(y0, y1 + 1):
                    cell = xi * p_num_Y + yi
                    for j in range(first_cell_particle[cell], first_cell_particle[cell + 1]):
                        id = cell_particle[j]
                        if i != id:
                            p2 = particle_pos[id]
                            d = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                            dist2 = np.dot(d, d)
                            if dist2 < min_dist2 and dist2 > 0.0:
                                dist = np.sqrt(dist2)
                                diff = (min_dist - dist) / dist * 0.5
                                p1[0] -= d[0] * diff
                                p1[1] -= d[1] * diff
                                p2[0] += d[0] * diff
                                p2[1] += d[1] * diff

                                particle_pos[i] = p1
                                particle_pos[id] = p2

    




def transferVelocities(toGrid : bool, flipRatio : float):
    global u, v, du, dv, prev_U, prev_V, cell_type, particle_pos, particle_vel
    global num_particles, f_num_X, f_num_Y, f_inv_spacing, h, f_num_cells

    n = f_num_Y
    # h = f_spacing
    h_1 = f_inv_spacing
    h_2 = 0.5 * h

    if(toGrid):
        prev_U = u
        prev_V = v
        du = [0.0] * f_num_cells
        dv = [0.0] * f_num_cells
        u = [0.0] * f_num_cells
        v = [0.0] * f_num_cells

        for i in range(f_num_cells):
            cell_type[i] = SOLID_CELL if s[i] == 0.0 else AIR_CELL

        for i in range(num_particles):
            x = particle_pos[i][0]
            y = particle_pos[i][1]
            xi = clamp(int(x * h_1), 0, f_num_X - 1)
            yi = clamp(int(y * h_1), 0, f_num_Y - 1)
            cell_num = xi * n + yi

            # print(cell_num, f_num_cells, len(cell_type))
            
            if (cell_type[cell_num] == AIR_CELL):
                cell_type[cell_num] = FLUID_CELL
                # cell_color[cell_num] = 1.0
            
    for component in [0,1]:
        dx = 0.0 if component == 0 else h_2
        dy = 0.0 if component == 1 else h_2

        f = u if component == 0 else v
        f_prev = prev_U if component == 0 else prev_V
        df = du if component == 0 else dv

        for i in range(num_particles):
            x = particle_pos[i][0]
            y = particle_pos[i][1]

            x = clamp(int(x * h_1), 0, p_num_X - 1)
            y = clamp(int(y * h_1), 0, p_num_Y - 1)

            x0 = min(int((x-dx) * h_1), p_num_X - 2)
            tx = ((x - dx) - x0 * h) * h_1
            x1 = min(x0 + 1, p_num_X - 2)

            y0 = min(int((y-dy) * h_1), p_num_Y - 2)
            ty = ((y - dy) - y0 * h) * h_1
            y1 = min(y0 + 1, p_num_Y - 2)

            sx = 1.0 - tx
            sy = 1.0 - ty

            d0 = sx*sy
            d1 = tx*sy
            d2 = tx*ty
            d3 = sx*ty

            nr0 = x0*n + y0
            nr1 = x1*n + y0
            nr2 = x1*n + y1
            nr3 = x0*n + y1

            if toGrid:
                pv = particle_vel[i][component]
                f[nr0] += d0 * pv
                f[nr1] += d1 * pv
                f[nr2] += d2 * pv
                f[nr3] += d3 * pv
                df[nr0] += d0
                df[nr1] += d1
                df[nr2] += d2
                df[nr3] += d3

            else:
                offset = n if component == 0 else 1
                valid0 = 1.0 if cell_type[nr0] != AIR_CELL or cell_type[nr0 - offset] != AIR_CELL else 0.0
                valid1 = 1.0 if cell_type[nr1] != AIR_CELL or cell_type[nr1 - offset] != AIR_CELL else 0.0
                valid2 = 1.0 if cell_type[nr2] != AIR_CELL or cell_type[nr2 - offset] != AIR_CELL else 0.0
                valid3 = 1.0 if cell_type[nr3] != AIR_CELL or cell_type[nr3 - offset] != AIR_CELL else 0.0

                vel = particle_vel[i][component]
                d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3
                
                if d > 0.0:
                    picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / d
                    corr = (valid0 * d0 * (f[nr0] - f_prev[nr0]) + valid1 * d1 * (f[nr1] - f_prev[nr1]) + valid2 * d2 * (f[nr2] - f_prev[nr2]) + valid3 * d3 * (f[nr3] - f_prev[nr3]) ) / d
                    flipV = vel + corr

                    particle_vel[i][component] = flipRatio * flipV + (1.0 - flipRatio) * picV

        if component == 0:
            u = f
            prev_U = f_prev
            du = df
        else:
            v = f    
            prev_V = f_prev
            dv = df

    if toGrid:
        for i in range(len(f)):
            if df[i] > 0.0:
                f[i] /= df[i]
        # restore solid cells
        for i in range(f_num_X):
            for j in range(f_num_Y):
                c = i * n + j
                if cell_type[c] == SOLID_CELL or (i > 0 and cell_type[c-n] == SOLID_CELL):
                    u[c] = prev_U[c]
                if cell_type[c] == SOLID_CELL or (j > 0 and cell_type[c-1] == SOLID_CELL):
                    v[c] = prev_V[c]

# Monte Carlo pressure gradient estimation probably goes somewhere in here or updateParticleDensity?
def solveIncmpressibility(num_iterations : int, dt : float, over_relaxation : float, compensateDrift : bool):
    global u, v, p, s, f_num_X, f_num_Y, f_inv_spacing, h, f_num_cells
    global prev_U, prev_V, rho, particle_density, particle_rest_density

    p = [0.0] * f_num_cells
    
    prev_U = u
    prev_V = v

    n = f_num_Y
    cp = rho * h / dt

    for it in range(num_iterations):
        for i in range(f_num_X):
            for j in range(f_num_Y):
                c = i * n + j
                if cell_type != FLUID_CELL:
                    continue

                left = (i - 1) * n + j
                right = (i + 1) * n + j
                bottom = i * n + j - 1
                top = i * n + j + 1

                # s0 = s[c]
                sx0 = s[left]
                sx1 = s[right]
                sy0 = s[bottom]
                sy1 = s[top]

                this_s = sx0 + sx1 + sy0 + sy1
                if this_s == 0.0:
                    continue

                div = u[right] - u[c] + v[top] - v[c]
                # div *= 1.9

                if(particle_rest_density > 0.0 and compensateDrift):
                    k = 1.0
                    compression = particle_density[c] - particle_rest_density
                    if compression > 0.0:
                        div = div - k * compression

                this_p = -div / this_s
                this_p *= over_relaxation
                p[c] += (this_p * cp)
                u[c] -= (this_p * sx0)
                u[right] += (this_p * sx1)
                v[c] -= (this_p * sy0)
                v[top] += (this_p * sy1)
                    
def updateParticleDensity():
    global particle_density, particle_rest_density, particle_pos, num_particles
    global f_num_X, f_num_Y, h, f_inv_spacing, f_num_cells, cell_type

    n = f_num_Y
    h_1 = f_inv_spacing
    h_2 = 0.5 * h

    d = [0.0] * len(particle_density)

    for i in range(num_particles):
        x = particle_pos[i][0]
        y = particle_pos[i][1]

        x = clamp(x, h, (f_num_X - 1) * h)
        y = clamp(y, h, (f_num_Y - 1) * h)

        x0 = int((x - h_2) * h_1)
        tx = ((x - h_2) - x0 * h) * h_1
        x1 = min(x0 + 1, f_num_X - 2)

        y0 = int((y - h_2) * h_1)
        ty = ((y - h_2) - y0 * h) * h_1
        y1 = min(y0 + 1, f_num_Y - 2)

        sx = 1.0 - tx
        sy = 1.0 - ty

        if (x0 < f_num_X and y0 < f_num_Y):
            d[x0 * n + y0] += sx * sy
        if (x1 < f_num_X and y0 < f_num_Y):
            d[x1 * n + y0] += tx * sy
        if (x1 < f_num_X and y1 < f_num_Y):
            d[x1 * n + y1] += tx * ty
        if (x0 < f_num_X and y1 < f_num_Y):
            d[x0 * n + y1] += sx * ty

    if(particle_rest_density == 0.0):
        sum = 0.0
        num_fluid_cells = 0
        for i in range(f_num_cells):
            if cell_type[i] == FLUID_CELL:
                sum += d[i]
                num_fluid_cells += 1
        if num_fluid_cells > 0:
                particle_rest_density = sum / num_fluid_cells
        print('rest density', particle_rest_density)

    particle_density = d

def simulate():
    integrateParticles(dt, gravity)
    pushParticlesApart(20)
    transferVelocities(True, 0.9)
    updateParticleDensity()
    solveIncmpressibility(100, dt, 1.9, False)
    transferVelocities(False, 0.9)
    # setBoundaryConditions()
    # updateParticles()



# Set solid cells
for i in range(f_num_X):
    for j in range(f_num_Y):
        c = i * f_num_Y + j
        x1 = [i * h, j * h]
        x2 = [(i + 1) * h, j * h]
        x3 = [(i + 1) * h, (j + 1) * h]
        x4 = [i * h, (j + 1) * h]
        # x - o dot n < 0
        c1_g = (x1[0] - ground_o[0]) * ground_n[0] + (x1[1] - ground_o[1]) * ground_n[1]
        c2_g = (x2[0] - ground_o[0]) * ground_n[0] + (x2[1] - ground_o[1]) * ground_n[1]
        c3_g = (x3[0] - ground_o[0]) * ground_n[0] + (x3[1] - ground_o[1]) * ground_n[1]
        c4_g = (x4[0] - ground_o[0]) * ground_n[0] + (x4[1] - ground_o[1]) * ground_n[1]
        
        c1_l = (x1[0] - left_o[0]) * left_n[0] + (x1[1] - left_o[1]) * left_n[1]
        c2_l = (x2[0] - left_o[0]) * left_n[0] + (x2[1] - left_o[1]) * left_n[1]
        c3_l = (x3[0] - left_o[0]) * left_n[0] + (x3[1] - left_o[1]) * left_n[1]
        c4_l = (x4[0] - left_o[0]) * left_n[0] + (x4[1] - left_o[1]) * left_n[1]

        c1_r = (x1[0] - right_o[0]) * right_n[0] + (x1[1] - right_o[1]) * right_n[1]
        c2_r = (x2[0] - right_o[0]) * right_n[0] + (x2[1] - right_o[1]) * right_n[1]
        c3_r = (x3[0] - right_o[0]) * right_n[0] + (x3[1] - right_o[1]) * right_n[1]
        c4_r = (x4[0] - right_o[0]) * right_n[0] + (x4[1] - right_o[1]) * right_n[1]

        s[c] = 1.0
        if c1_l < 0.0 or c2_l < 0.0 or c3_l < 0.0 or c4_l < 0.0:
            s[c] = 0.0 
        if c1_g < 0.0 or c2_g < 0.0 or c3_g < 0.0 or c4_g < 0.0:
            s[c] = 0.0
        if c1_r < 0.0 or c2_r < 0.0 or c3_r < 0.0 or c4_r < 0.0:
            s[c] = 0.0

# simulation with visualization
resolution = np.array([600, 600])
offset = [0.0,0.0]
scale = max(resolution / [width, height])
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]


time_step = 0
draw_grid = False
draw_cells = False
show_numbers = False
calc_KE = False
particles.write_to_file(time_step, particle_pos, num_particles)
screen = pygame.display.set_mode(resolution)
# display_surface = pygame.display.set_mode((width, height))
running = True
paused = False
while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                draw_grid = not draw_grid
            if event.key == pygame.K_c:
                draw_cells = not draw_cells
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_n:
                show_numbers = not show_numbers
            if event.key == pygame.K_k:
                calc_KE = not calc_KE

    if paused:
        pygame.time.wait(int(dt * 1000))
        continue
    
    print('### Time step', time_step, '###')

    # fill the background and draw the square
    screen.fill((255, 255, 255))

    #draw grid
    if(draw_cells):
        for i in range(f_num_X):
            for j in range(f_num_Y):
                c = i * f_num_Y + j
                if cell_type[c] == AIR_CELL:
                    # print('air cell', c)
                    pygame.draw.rect(screen, (255, 255, 255), (screen_projection([i * h, (j+1) * h]), [scale * h, scale *h]))
                elif cell_type[c] == FLUID_CELL:
                    # print('fluid cell', c)
                    pygame.draw.rect(screen, (20, 20, 170), (screen_projection([i * h, (j+1) * h]), [scale * h, scale *h]))
                elif cell_type[c] == SOLID_CELL:
                    # print('solid cell', c)
                    pygame.draw.rect(screen, (0, 150, 0), (screen_projection([i * h, (j+1) * h]), [scale * h, scale * h]))
                
        
    
    if(draw_grid):
        for i in range(f_num_X):
            pygame.draw.aaline(screen, (80, 80, 0), screen_projection([i * h, 0]), screen_projection([i * h, height]))
        for j in range(f_num_Y):
            pygame.draw.aaline(screen, (80, 80, 0), screen_projection([0, j * h]), screen_projection([width, j * h]))
    if(show_numbers):        
        for i in range(f_num_X):
            for j in range(f_num_Y):
                c = i * f_num_Y + j
                text = pygame.font.Font(None, 20).render(str(c), True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (screen_projection([(i+ 0.5) * h, (j + 0.5) * h]))
                screen.blit(text, textRect)

    # pygame.draw.circle(screen, (255, 0, 0), screen_projection([0, 0]), 2 * scale)  # draw a red circle
    pygame.draw.aaline(screen, (255, 0, 255), screen_projection([0,ground_o[1]]), screen_projection([width, ground_o[1]]))   # ground 
    pygame.draw.aaline(screen, (255, 0, 255), screen_projection([left_o[0], 0]), screen_projection([left_o[0], height]))   # left wall
    pygame.draw.aaline(screen, (255, 0, 255), screen_projection([right_o[0], 0]), screen_projection([right_o[0], height]))   # right wall

    # draw particles
    for xId in range(0, num_particles):
        xI = particle_pos[xId]
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), particle_radius * scale)

    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    simulate()
    if(calc_KE):
        ke = energy.kineticEnergy(particle_mass, particle_vel)
        print('Kinetic energy:', ke)

    time_step += 1
    pygame.time.wait(int(dt * 1000))
    particles.write_to_file(time_step, particle_pos, num_particles)

pygame.quit()