# (c) Yann Thorimbert 2017
"""This example module shows how to generate fast 2D terrain or noise with
numpy, for Python2 and Python3.

Since this aims at performance, the code is more complex that necessary for
understanding noise. One should use the pure python version if the performance
is not crucial.
"""
from __future__ import print_function, division
import numpy as np
from numpy.polynomial.polynomial import polyval2d
import thornoise.numpygen.colorscale as colorscale
import thornoise.numpygen.noisegen as ng

if __name__ == "__main__":
    # First we choose the type of noise
##    c = cache.ZeroGradient() #fastest
##    c = cache.Generic() #prettiest ?
    c = ng.Perlin() #most famous
##    c.DEPTH = 6
##    c.MIN_N = 4
##    c.S = 12
##    c.DOM_DIVIDER = 2 #if change, should change S and min_n
##    c.H_DIVIDER = 1.7
##    c.WORLD_SIZE = (4,4) #size in number of chunks. The world is a torus.
    c.build()
    #
    colormap = colorscale.SUMMER #How height is transformed into color
    SEED = 0
    init_chunk = (0,230) #any couple of positive integers
    chunk = np.array(init_chunk)%c.WORLD_SIZE #dont go further than world limit
    chunk = tuple(init_chunk)
    #
    hmap = ng.generate_terrain(chunk, c) #generate actual data
    hmap = ng.normalize(hmap) #scales to [0,1] range
    cmap = colormap.get(hmap) #array of colors
    # Visualization with PYGAME:
    HAS_PYGAME = False
    try:
        import pygame
        from pygame import gfxdraw
        HAS_PYGAME = True
    except:
        print("Pygame not found on this distribution! However, "+\
              "you can use this module's 'generate_terrain' functions "+\
              "to generate terrain or noise in a 2D array.")
    if HAS_PYGAME:
        pygame.init()
        screen = pygame.display.set_mode((c.S,c.S))
        surface = pygame.surfarray.make_surface(cmap) #convert to surface using cmap
        screen.blit(surface,(0,0)) #draw on screen
        pygame.display.flip() #refresh screen
        #
        looping = True
        while looping:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    looping = False
        pygame.quit()
