import numpy as np  # numpy for numerical operations
import pygame       # pygame for visualization
pygame.init()

import simulator as sim  # simulation for the simulation class
import particles       # particles for the particle class


# simulation parameters
dt = sim.dt
max_particles = sim.max_particles
width = sim.width
height = sim.height
particle_radius = sim.particle_radius

particle_pos = np.zeros((max_particles, 2))  # particle positions

# simulation with visualization
resolution = np.array([600, 600])
offset = [0.0,0.0]
scale = max(resolution / [width, height])
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]


time_step = 0
max_time_step = sim.max_time_step
num_particles = 0
screen = pygame.display.set_mode(resolution)
# display_surface = pygame.display.set_mode((width, height))
running = True
paused = False
while running and __name__ == "__main__":
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_q:
                running = False

    if paused:
        pygame.time.wait(int(dt * 1000))
        continue
    
    print('### Time step', time_step, '###')

    # get timestep frame

    [particle_pos, num_particles, reset] = particles.read_from_file(time_step)
    if reset:
        time_step = 0
        continue

    # fill the background and draw the square
    screen.fill((255, 255, 255))

    # pygame.draw.circle(screen, (255, 0, 0), screen_projection([0, 0]), 2 * scale)  # draw a red circle
    # pygame.draw.aaline(screen, (255, 0, 255), screen_projection([0,ground_o[1]]), screen_projection([width, ground_o[1]]))   # ground 
    # pygame.draw.aaline(screen, (255, 0, 255), screen_projection([left_o[0], 0]), screen_projection([left_o[0], height]))   # left wall
    # pygame.draw.aaline(screen, (255, 0, 255), screen_projection([right_o[0], 0]), screen_projection([right_o[0], height]))   # right wall

    # draw particles
    for xId in range(0, num_particles):
        xI = particle_pos[xId]
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), particle_radius * scale)

    # draw time step as text
    # font = pygame.font.Font(None, 36)
    # text = font.render(f"Time step: {time_step}", True, (0, 0, 0))
    # screen.blit(text, (10, 10))


    pygame.display.flip()   # flip the display
    
    # simulate()
    # if(calc_KE):
    #     ke = energy.kineticEnergy(particle_mass, particle_vel)
    #     print('Kinetic energy:', ke)

    time_step += 1
    if time_step > max_time_step:
        time_step = 0 
    pygame.time.wait(int(dt * 1000))

pygame.quit()
