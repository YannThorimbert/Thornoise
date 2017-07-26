# (c) Yann Thorimbert 2017
"""This module provides fast functions to generate 2D terrain or noise with
numpy, for Python2 and Python3.

Since this aims at performance, the code is more complex that necessary for
understanding noise. One should use the pure python version if the performance
is not crucial.
"""
from __future__ import print_function, division
import pygame, time
import numpy as np
from pygame import gfxdraw
from numpy.polynomial.polynomial import polyval2d
import pygame.surfarray
import colorscale

#Scale paramters (impact on fractal behaviour)
DEPTH = 6 #number of scales ("octaves" in Perlin's nomenclature)
H_DIVIDER = 2. #height divider (~"1/persistance" in Perlin's nomenclature)
H_DIVIDER_0 = 1. #initial height value (at max level)
DOM_DIVIDER = 2. #domain divider (~"lacunarity" in Perlin's nomenclature)
DOM_DIVIDER_0 = 1. #initial domain divider value
DEEP_CONSTANT = 4 #max_n/2**depth = constant. change "zoom".

#Derived parameters
LEVELS = range(DEPTH)
MAX_N = DEEP_CONSTANT*2**DEPTH #1/min_frequency
APPROX_SCREEN_W = max(512, MAX_N) #automatically set a suitable resolution
S = (APPROX_SCREEN_W // MAX_N) * MAX_N #so S is a multiple of MAX_N

#Pre-compute different parameters for each scale
PARAM_N, PARAM_H = [], []
i_dom_divider = DOM_DIVIDER_0
i_h = H_DIVIDER_0
for i in LEVELS:
    i_dom_divider *= DOM_DIVIDER
    PARAM_N.append(int(MAX_N/i_dom_divider)) #number of domains at level i
    i_h *= H_DIVIDER
    PARAM_H.append(i_h)
PARAM_N = PARAM_N[::-1] #invert list
PARAM_H = PARAM_H[::-1]
RES = []
for k in LEVELS:
    RES.append(int(S / PARAM_N[k])) #resolution of domain at level k

def _get_x(k): #used for builing cache (see below)
    res = RES[k]
    domain = np.arange(0., 1., 1./res)
    a = np.zeros((res,res))
    for x in range(res):
        a[x,:] = domain[x]
    return a

##print("Start building cache")
#Lines below pre-compute space evaluation for fast dynamic generation.
#Could be much more optimized, but we don't care it's not in the loop.
SMOOTHSTEP_X = []
SMOOTHSTEP_Y = []
PERLIN_SMOOTHSTEP_X = []
PERLIN_SMOOTHSTEP_Y = []
CACHE_XY = []
X = []
Y = []
XM1 = []
YM1 = []
CACHE_XiYj = []
for k in LEVELS:
##    print("-", end="")
    x = _get_x(k)
    y = x.T
    X.append(x)
    XM1.append(x - 1.)
    Y.append(y)
    YM1.append(y - 1.)
    SMOOTHSTEP_X.append(3.*x**2 - 2.*x**3)
    SMOOTHSTEP_Y.append(3.*y**2 - 2.*y**3)
    PERLIN_SMOOTHSTEP_X.append(6.*x**5 - 15.*x**4 + 10.*x**3)
    PERLIN_SMOOTHSTEP_Y.append(6.*y**5 - 15.*y**4 + 10.*y**3)
    CACHE_XY.append(x*y - 3.*(y*x**2 + x*y**2) + 2.*(y*x**3 + x*y**3))
    dictij = {}
    for i in range(4):
        for j in range(4):
            dictij[(i,j)] = x**i * y**j
    CACHE_XiYj.append(dictij)
##print("> Cache finished")

def RandArray(c, n): #return rand array with values comprised in [0, n[
    return c*(2*np.random.random((n,n)) - 1)

def _set_seeded_condition(l, t, a, n, val, flag):
    #lines (can be optimized, corners don't need to be set here...)
    right = (l+1)%WORLD_SIZE[0]
    bottom = (t+1)%WORLD_SIZE[1]
    np.random.seed((l,t,n,flag,0)) #left
    a[0,:] = val*(2*np.random.random(n+1) - 1)
    np.random.seed((right,t,n,flag,0)) #right
    a[n,:] = val*(2*np.random.random(n+1) - 1)
    np.random.seed((l,t,n,flag,1)) #top
    a[:,0] = val*(2*np.random.random(n+1) - 1)
    np.random.seed((l,bottom,n,flag,1)) #bottom
    a[:,n] = val*(2*np.random.random(n+1) - 1)
    #corners
    np.random.seed((l,t,n,flag)) #topleft
    a[0,0] = val*(2*np.random.random() - 1)
    np.random.seed((right,t,n,flag)) #topright
    a[n,0] = val*(2*np.random.random() - 1)
    np.random.seed((l,bottom,n,flag)) #bottomleft
    a[0,n] = val*(2*np.random.random() - 1)
    np.random.seed((right,bottom,n,flag)) #bottomright
    a[n,n] = val*(2*np.random.random() - 1)

def get_seeded_conditions(truechunk, k):
    """This function (along with _set_seeded_condition) is used in order to
    guaranty that the produced data will always be the same for a given position
    in space and for a given seed.
    A really minimalist code could just use pure random array instead of the
    returned arrays.
    It is really not optimal to use this version for D2M1N3, as only <tabh> is
    used, and other arrays are ignored."""
    n = PARAM_N[k]
    h = PARAM_H[k]
    p = 1.
    l,t = truechunk
    np.random.seed([l,t,n,0+SEED]) #bulk
    tabh,tabf,tabg = RandArray(h,n+1),RandArray(p,n+1),RandArray(p,n+1)
    _set_seeded_condition(l,t,tabh,n,h,1+SEED)
    _set_seeded_condition(l,t,tabf,n,p,2+SEED)
    _set_seeded_condition(l,t,tabg,n,p,3+SEED)
    return tabh, tabf, tabg

def generate_terrain(chunk):
    """Returns a <S> times <S> array of heigth values using <chunk> as seed.
    """
    hmap = np.zeros((S,S))
    for k in LEVELS:
        h,f,g = get_seeded_conditions(chunk, k)
        for x in range(PARAM_N[k]):
            for y in range(PARAM_N[k]):
                p = POLYNOM(h[x:x+2,y:y+2], f[x:x+2,y:y+2], g[x:x+2,y:y+2])
                p.fill_array(hmap, k, x, y)
    return hmap


class PolynomZG: #zero-gradient D2M1N3 polynom (see article)

    def __init__(self, h, f, g): #h is a 2*2 array containing imposed heights
        self.h0 = h[0,0]
        self.dhx = h[1,0]-self.h0
        self.dhy = h[0,1]-self.h0
        self.A = self.dhx-h[1,1]+h[0,1]

    def domain_eval(self, k): #use cached space for fast evaluation
        result = self.dhx*SMOOTHSTEP_X[k] +\
                 self.dhy*SMOOTHSTEP_Y[k] +\
                 self.A*CACHE_XY[k] +\
                 self.h0
        return result

    def fill_array(self, a, k, x0, y0): #fill array a with values of self
        res = RES[k]
        x0 *= res
        y0 *= res
        a[x0:x0+res,y0:y0+res] += self.domain_eval(k)

class PolynomGeneric(PolynomZG):

    def __init__(self, h, f, g):
        A = h[0,1] + h[1,0] - h[0,0] - h[1,1]
        self.c = np.zeros((4,4))
        #
        self.c[0,0] = h[0,0]
        self.c[1,0] = f[0,0]
        self.c[0,1] = g[0,0]
        #
        self.c[2,0] = 3.*(h[1,0]-h[0,0]) - 2.*f[0,0] - f[1,0]
        self.c[0,2] = 3.*(h[0,1]-h[0,0]) - 2.*g[0,0] - g[0,1]
        self.c[3,0] = f[1,0] + f[0,0] - 2.*(h[1,0]-h[0,0])
        self.c[0,3] = g[0,1] + g[0,0] - 2.*(h[0,1]-h[0,0])
        self.c[1,1] = A + g[1,0] + f[0,1] - g[0,0] - f[0,0]# + self.c[2,2]
        #
        self.c[3,1] = f[1,1] + f[0,1] - 2.*(h[1,1]-h[0,1]) - self.c[3,0]
        self.c[1,3] = g[1,1] + g[1,0] - 2.*(h[1,1]-h[1,0]) - self.c[0,3]
        #
        self.c[2,1] = 3.*(h[1,1]-h[0,1]) - 2.*f[0,1] - f[1,1] - self.c[2,0]
        self.c[1,2] = 3.*(h[1,1]-h[1,0]) - 2.*g[1,0] - g[1,1] - self.c[0,2]

    def domain_eval(self, k): #! k = k. zoom level rajoute a l'appel de fonction
        result = self.c[0,0] + \
                 self.c[1,0]*CACHE_XiYj[k][(1,0)] +\
                 self.c[2,0]*CACHE_XiYj[k][(2,0)] +\
                 self.c[3,0]*CACHE_XiYj[k][(3,0)] +\
                 self.c[0,1]*CACHE_XiYj[k][(0,1)] +\
                 self.c[0,2]*CACHE_XiYj[k][(0,2)] +\
                 self.c[0,3]*CACHE_XiYj[k][(0,3)] +\
                 self.c[1,1]*CACHE_XiYj[k][(1,1)] +\
                 self.c[2,1]*CACHE_XiYj[k][(2,1)] +\
                 self.c[3,1]*CACHE_XiYj[k][(3,1)] +\
                 self.c[1,2]*CACHE_XiYj[k][(1,2)] +\
                 self.c[1,3]*CACHE_XiYj[k][(1,3)]
        return result

class PolynomPerlin(PolynomZG):

    def __init__(self, h, f, g):
        self.f = f
        self.g = g

    def domain_eval(self, k):
        topleft = self.f[0,0]*X[k] + self.g[0,0]*Y[k] #topleft corner's height contribution
        topright = self.f[1,0]*XM1[k]+self.g[1,0]*Y[k]
        bottomleft = self.f[0,1]*X[k] + self.g[0,1]*YM1[k]
        bottomright = self.f[1,1]*XM1[k] + self.g[1,1]*YM1[k]
        #
        htop = topleft + PERLIN_SMOOTHSTEP_X[k] * (topright - topleft) #height along x, for y = 0
        hbottom = bottomleft + PERLIN_SMOOTHSTEP_X[k] * (bottomright - bottomleft) #... for y = 1.
        hmiddle = htop + PERLIN_SMOOTHSTEP_Y[k] * (hbottom-htop) #for y = y
        return hmiddle

    def fill_array(self, a, k, x0, y0):
        res = RES[k]
        x0 *= res
        y0 *= res
        a[x0:x0+res,y0:y0+res] += self.domain_eval(k) * PARAM_H[k]

def normalize(hmap):
    minh, maxh = np.min(hmap), np.max(hmap)
    return (hmap-minh)/(maxh-minh)

if __name__ == "__main__":
    # First we choose the type of noise
##    POLYNOM = PolynomZG #fastest
##    POLYNOM = PolynomGeneric #prettiest ?
    POLYNOM = PolynomPerlin #most famous
    colormap = colorscale.SUMMER #How height is transformed into color
    SEED = 0
    WORLD_SIZE = (4,4) #size in number of chunks. The world is a torus here.
    init_chunk = (0,0) #any couple of positive integers
    chunk = np.array(init_chunk)%WORLD_SIZE
    chunk = tuple(init_chunk)
    hmap = generate_terrain(chunk) #generate actual data
    hmap = normalize(hmap) #scales to [0,1] range
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
        screen = pygame.display.set_mode((S,S))
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
