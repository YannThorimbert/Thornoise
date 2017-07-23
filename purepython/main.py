# (c) Yann Thorimbert 2017
"""This example module shows how to generate 2D terrain or noise with
no dependency, for Python2 and Python3."""
from __future__ import print_function
import random, math
import noisegen as ng

if __name__ == "__main__":
    # 1. We define some constants
    random.seed(14000000) #choose a seed
    S = 256 #Resolution of generated terrain
    MAX_OCTAVES = int(math.log(S,2))
    N_OCTAVES = MAX_OCTAVES #you can choose to use less (but not more) octaves
    SMOOTH, DTERM, IDX = ng.get_cache(MAX_OCTAVES, S)
    summer =  ng.ColorScale([  [(0,0,0), (0,0,100), 0.],            #0. deep
                         [(0,0,100), (0,30,255), 0.52],             #1. shallow
                         [(0,30,255), (137, 131, 200), 0.597],      #2. sand
                         [(137, 131, 200), (237, 201, 175), 0.6],   #3. sand
                         [(237, 201, 175), (50,85,10), 0.605],      #4. sand
                         [(50,85,10), (50,180,50), 0.78],           #5. forest
                         [(50,180,50),(150,180,150), 0.85],
                         [(150,180,150), (255,255,255), 1.000001],    #6. snow
                         [(255,255,255), (255,255,255), 10.]],      #7. snow
                         minval = -10.)
    # We generate the actual terrain or noise (here we use all arguments)
    terrain = ng.generate_terrain(n_octaves=MAX_OCTAVES,
                                    size=S,
                                    chunk=(0,0),
                                    persistance=2.)
    #generation tiwh default arguments, and with other methods
    #note that each generation method takes the same arguments
##    terrain = ng.generate_terrain_cache(MAX_OCTAVES, S)
##    terrain = ng.generate_terrain_local(MAX_OCTAVES, S)
    ng.normalize(terrain)
    # Visualization with PYGAME:
    HAS_PYGAME = False
    try:
        import pygame
        from pygame import gfxdraw
        HAS_PYGAME = True
    except:
        print("Pygame not found on this distribution! However, "+\
              "you can use this module's 'generate_terrain_xxx' functions "+\
              "to generate terrain or noise in a 2D array.")
    if HAS_PYGAME:
        screen = pygame.display.set_mode((S,S))
        s = ng.build_surface(terrain, summer) #we build the surface
        screen.blit(s,(0,0))
        pygame.display.flip()
        #
        stay = True
        while stay:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    stay = False
        pygame.quit()

