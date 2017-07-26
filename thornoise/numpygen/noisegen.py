# (c) Yann Thorimbert 2017
"""This module provides fast functions to generate 2D terrain or noise with
numpy, for Python2 and Python3.

Since this aims at performance, the code is more complex that necessary for
understanding noise. One should use the pure python version if the performance
is not crucial.
"""
from __future__ import print_function, division
import numpy as np
from numpy.polynomial.polynomial import polyval2d
import thornoise.numpygen.colorscale as colorscale

###Scale paramters (impact on fractal behaviour)
##DEPTH = 6 #number of scales ("octaves" in Perlin's nomenclature)
##H_DIVIDER = 2. #height divider (~"1/persistance" in Perlin's nomenclature)
##H_DIVIDER_0 = 1. #initial height value (at max level)
##DOM_DIVIDER = 2. #domain divider (~"lacunarity" in Perlin's nomenclature)
##DOM_DIVIDER_0 = 1. #initial domain divider value
##DEEP_CONSTANT = 4 #max_n/2**depth = constant. change "zoom".
##
##class NoiseParameters:
##
##    def __init__(self, depth=DEPTH, hdivider=H_DIVIDER, hdivider0=H_DIVIDER_0,
##                 domdivider=DOM_DIVIDER, domdivider0=DOM_DIVIDER_0,
##                 deep_constant=DEEP_CONSTANT):
##        self.depth = depth
##        self.hdivider = hdivider
##        self.hdivider0 = hdivider0
##        self.domdivider = domdivider
##        self.domdivider0 = domdivider0
##        self.deep_constant = deep_constant
##        #to be computed into refresh():
##        self.levels = None
##        self.max_n = None
##        self.S = None
##        self.refresh()
##
##    def refresh(self):
##        #Derived parameters
##        self.levels = range(self.depth)
##        self.max_n = self.deep_constant*2**self.depth
##        approx_S = max(512, MAX_N) #automatically set a suitable resolution
##        #so S is a multiple of MAX_N:
##        self.S = (APPROX_SCREEN_W // self.max_n) * self.max_n
##
##    def compute_cache(self):
##        #Pre-compute different parameters for each scale (compute cached values)
##        PARAM_N, PARAM_H = [], []
##        i_dom_divider = p.domdivider_0
##        i_h = p.hdivider0
##        for i in LEVELS:
##            i_dom_divider *= p.domdivider
##            PARAM_N.append(int(p.max_n/i_dom_divider)) #number of domains at level i
##            i_h *= p.hdivider
##            PARAM_H.append(i_h)
##        PARAM_N = PARAM_N[::-1] #invert list
##        PARAM_H = PARAM_H[::-1]
##        RES = []
##        for k in LEVELS:
##            RES.append(int(p.S / PARAM_N[k])) #resolution of domain at level k
##
##
##p = NoiseParameters() #look at the beginning of this script to change noise params
##p.refresh()
##
##
##def _get_x(k): #used for builing cache (see below)
##    res = RES[k]
##    domain = np.arange(0., 1., 1./res)
##    a = np.zeros((res,res))
##    for x in range(res):
##        a[x,:] = domain[x]
##    return a
##
####print("Start building cache")
###Lines below pre-compute space evaluation for fast dynamic generation.
###Could be much more optimized, but we don't care it's not in the loop.
##SMOOTHSTEP_X = []
##SMOOTHSTEP_Y = []
##PERLIN_SMOOTHSTEP_X = []
##PERLIN_SMOOTHSTEP_Y = []
##CACHE_XY = []
##X = []
##Y = []
##XM1 = []
##YM1 = []
##CACHE_XiYj = []
##for k in LEVELS:
####    print("-", end="")
##    x = _get_x(k)
##    y = x.T
##    X.append(x)
##    XM1.append(x - 1.)
##    Y.append(y)
##    YM1.append(y - 1.)
##    SMOOTHSTEP_X.append(3.*x**2 - 2.*x**3)
##    SMOOTHSTEP_Y.append(3.*y**2 - 2.*y**3)
##    PERLIN_SMOOTHSTEP_X.append(6.*x**5 - 15.*x**4 + 10.*x**3)
##    PERLIN_SMOOTHSTEP_Y.append(6.*y**5 - 15.*y**4 + 10.*y**3)
##    CACHE_XY.append(x*y - 3.*(y*x**2 + x*y**2) + 2.*(y*x**3 + x*y**3))
##    dictij = {}
##    for i in range(4):
##        for j in range(4):
##            dictij[(i,j)] = x**i * y**j
##    CACHE_XiYj.append(dictij)
####print("> Cache finished")

def RandArray(c, n): #return rand array with values comprised in [0, n[
    return c*(2*np.random.random((n,n)) - 1)

def _set_seeded_condition(l, t, a, n, val, flag, ws):
    #lines (can be optimized, corners don't need to be set here...)
    right = (l+1)%ws[0]
    bottom = (t+1)%ws[1]
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

def get_seeded_conditions(truechunk, k, c):
    """This function (along with _set_seeded_condition) is used in order to
    guaranty that the produced data will always be the same for a given position
    in space and for a given seed.
    A really minimalist code could just use pure random array instead of the
    returned arrays.
    It is really not optimal to use this version for D2M1N3, as only <tabh> is
    used, and other arrays are ignored."""
    n = c.PARAM_N[k]
    h = c.PARAM_H[k]
    p = 1.
    l,t = truechunk
    np.random.seed([l,t,n,0]) #bulk
    tabh,tabf,tabg = RandArray(h,n+1),RandArray(p,n+1),RandArray(p,n+1)
    _set_seeded_condition(l,t,tabh,n,h,1,c.WORLD_SIZE)
    _set_seeded_condition(l,t,tabf,n,p,2,c.WORLD_SIZE)
    _set_seeded_condition(l,t,tabg,n,p,3,c.WORLD_SIZE)
    return tabh, tabf, tabg

def generate_terrain(chunk, c):
    """Returns an of heigth values using <chunk> as seed and <p> as parameters.
    """
    hmap = np.zeros((c.S,c.S))
    for k in c.LEVELS:
        h,f,g = get_seeded_conditions(chunk, k, c)
        for x in range(c.PARAM_N[k]):
            for y in range(c.PARAM_N[k]):
                pol = c.polynom(h[x:x+2,y:y+2], f[x:x+2,y:y+2], g[x:x+2,y:y+2])
                pol.fill_array(hmap, c, k, x, y)
    return hmap

def normalize(hmap):
    minh, maxh = np.min(hmap), np.max(hmap)
    return (hmap-minh)/(maxh-minh)


def get_x(res):
    domain = np.arange(0., 1., 1./res)
    a = np.zeros((res,res))
    for x in range(res):
        a[x,:] = domain[x]
    return a

def s1(x):
    return x

def s3(x):
    return 3.*x**2 - 2.*x**3

def s5(x):
    return 6.*x**5 - 15.*x**4 + 10.*x**3

def s7(x):
    return -20.*x**7 + 70.*x**6 -84.*x**5 + 35.*x**4

def s9(x):
    return 70*x**9 -315*x**8 + 540*x**7 -420*x**6 + 126*x**5

smoothstep = {1:s1, 3:s3, 5:s5, 7:s7, 9:s9}


class PolynomZG: #zero-gradient D2M1N3 polynom (see article)

    def __init__(self, h, f, g): #h is a 2*2 array containing imposed heights
        self.h0 = h[0,0]
        self.dhx = h[1,0]-self.h0
        self.dhy = h[0,1]-self.h0
        self.A = self.dhx-h[1,1]+h[0,1]

    def domain_eval(self, c, k): #use cached space for fast evaluation
        result = self.dhx*c.SMOOTHSTEP_X[k] +\
                 self.dhy*c.SMOOTHSTEP_Y[k] +\
                 self.A*c.XY[k] +\
                 self.h0
        return result

    def fill_array(self, a, c, k, x0, y0): #fill array a with values of self
        res = c.RES[k]
        x0 *= res
        y0 *= res
        a[x0:x0+res,y0:y0+res] += self.domain_eval(c, k)

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

    def domain_eval(self, c, k): #! k = k. zoom level rajoute a l'appel de fonction
        result = self.c[0,0] + \
                 self.c[1,0]*c.XiYj[k][(1,0)] +\
                 self.c[2,0]*c.XiYj[k][(2,0)] +\
                 self.c[3,0]*c.XiYj[k][(3,0)] +\
                 self.c[0,1]*c.XiYj[k][(0,1)] +\
                 self.c[0,2]*c.XiYj[k][(0,2)] +\
                 self.c[0,3]*c.XiYj[k][(0,3)] +\
                 self.c[1,1]*c.XiYj[k][(1,1)] +\
                 self.c[2,1]*c.XiYj[k][(2,1)] +\
                 self.c[3,1]*c.XiYj[k][(3,1)] +\
                 self.c[1,2]*c.XiYj[k][(1,2)] +\
                 self.c[1,3]*c.XiYj[k][(1,3)]
        return result

    def fill_array(self, a, c, k, x0, y0): #fill array a with values of self
        res = c.RES[k]
        x0 *= res
        y0 *= res
        a[x0:x0+res,y0:y0+res] += self.domain_eval(c, k) * c.PARAM_H[k]

class PolynomPerlin(PolynomZG):

    def __init__(self, h, f, g):
        self.f = f
        self.g = g

    def domain_eval(self, c, k):
        topleft = self.f[0,0]*c.X[k] + self.g[0,0]*c.Y[k] #topleft corner's height contribution
        topright = self.f[1,0]*c.XM1[k]+self.g[1,0]*c.Y[k]
        bottomleft = self.f[0,1]*c.X[k] + self.g[0,1]*c.YM1[k]
        bottomright = self.f[1,1]*c.XM1[k] + self.g[1,1]*c.YM1[k]
        #
        htop = topleft + c.SMOOTHSTEP_X[k] * (topright - topleft) #height along x, for y = 0
        hbottom = bottomleft + c.SMOOTHSTEP_X[k] * (bottomright - bottomleft) #... for y = 1.
        hmiddle = htop + c.SMOOTHSTEP_Y[k] * (hbottom-htop) #for y = y
        return hmiddle

    def fill_array(self, a, c, k, x0, y0):
        res = c.RES[k]
        x0 *= res
        y0 *= res
        a[x0:x0+res,y0:y0+res] += self.domain_eval(c,k) * c.PARAM_H[k]

class Cache:
    name = "Abstract cache"

    def __init__(self):
        #Scale paramters (impact on fractal behaviour)
        self.DEPTH = 7#number of levels (number of scales)
        self.H_DIVIDER = 2. #height divider (~"1/persistance" in Perlin's nomenclature)
        self.DOM_DIVIDER = 2 #domain divider (~"lacunarity" in Perlin's nomenclature)
        self.MIN_N = 1
        self.S = 512 #chunk size
        #
        self.LEVELS = None
        self.PARAM_N, PARAM_H = None, None
        self.RES = None
        #
        self.X = None
        self.Y = None
        #
        self.WORLD_SIZE = (1,1)

    def build_params(self):
        #Derived parameters
        self.LEVELS = range(self.DEPTH)
        #Pre-compute different parameters for each scale
        self.PARAM_N, self.PARAM_H = [], []
        self.RES = []
        n = self.MIN_N
        for i in self.LEVELS:
            self.PARAM_N.append(n) #number of domains at level i
            self.PARAM_H.append(1./self.H_DIVIDER**i)
            assert self.S%n == 0
            self.RES.append(int(self.S / n)) #resolution of domain at level k
            n *= self.DOM_DIVIDER
        assert self.MIN_N > 0
        assert self.PARAM_N[-1] <= self.S

##        for i in self.LEVELS:
##            self.PARAM_N.append(int(self.MAX_N/i_dom_divider)) #number of domains at level i
##            i_dom_divider *= self.DOM_DIVIDER
##            self.PARAM_H.append(i_h)
##            i_h *= self.H_DIVIDER
##        self.PARAM_N = self.PARAM_N[::-1] #invert list
##        self.PARAM_H = self.PARAM_H[::-1]
##        self.RES = []
##        for k in self.LEVELS:
##            self.RES.append(int(self.S / self.PARAM_N[k])) #resolution of domain at level k

    def get_x(self,k):
        return get_x(self.RES[k])
##    def get_x(self, k): #used for builing cache (see below)
##        res = self.RES[k]
##        domain = np.arange(0., 1., 1./res)
##        a = np.zeros((res,res))
##        for x in range(res):
##            a[x,:] = domain[x]
##        return a

    def build_cache(self):
        self.X, self.Y = [], []
        for k in self.LEVELS:
            x = self.get_x(k)
            self.X.append(x)
            self.Y.append(x.T)

    def build(self):
        self.build_params()
        print("Start building cache:", self.name, end="")
        self.build_cache()
        print("... cache built.")

class ZeroGradient(Cache):
    name = "ZG cache"
    polynom = PolynomZG

    def __init__(self):
        Cache.__init__(self)
        self.sdegree = 3
        self.SMOOTHSTEP_X = None
        self.SMOOTHSTEP_Y = None
        self.XY = None

    #Lines below pre-compute space evaluation for fast dynamic generation.
    #Could be much more optimized, but we don't care it's not in the loop.
    def build_cache(self):
        Cache.build_cache(self)
        self.SMOOTHSTEP_X, self.SMOOTHSTEP_Y, self.XY = [], [], []
        for k in self.LEVELS:
            x,y = self.X[k], self.Y[k] #aliases for short notation
            sx = smoothstep[self.sdegree](x)
            sy = sx.T
            self.SMOOTHSTEP_X.append(sx)
            self.SMOOTHSTEP_Y.append(sy)
            self.XY.append(x*y - y*sx - x*sy)

class Generic(Cache):
    name = "Generic D2M1N3 cache"
    polynom = PolynomGeneric

    def __init__(self):
        Cache.__init__(self)
        self.XiYj = None

    #Lines below pre-compute space evaluation for fast dynamic generation.
    #Could be much more optimized, but we don't care it's not in the loop.
    def build_cache(self):
        Cache.build_cache(self)
        self.XiYj = []
        for k in self.LEVELS:
            x,y = self.X[k], self.Y[k] #aliases for short notation
            dictij = {}
            for i in range(4):
                for j in range(4):
                    dictij[(i,j)] = (x**i) * (y**j)
            self.XiYj.append(dictij)


class Perlin(Cache):
    name = "Perlin cache"
    polynom = PolynomPerlin

    def __init__(self):
        Cache.__init__(self)
        self.sdegree = 5
        self.SMOOTHSTEP_X = None
        self.SMOOTHSTEP_Y = None
        self.XM1 = None
        self.YM1 = None

    #Lines below pre-compute space evaluation for fast dynamic generation.
    #Could be much more optimized, but we don't care it's not in the loop.
    def build_cache(self):
        Cache.build_cache(self)
        self.SMOOTHSTEP_X, self.SMOOTHSTEP_Y = [], []
        self.XM1, self.YM1 = [], []
        for k in self.LEVELS:
            x,y = self.X[k], self.Y[k] #aliases for short notation
            sx = smoothstep[self.sdegree](x)
##            sx = 3.*x**2 - 2.*x**3
            sy = sx.T
            self.SMOOTHSTEP_X.append(sx)
            self.SMOOTHSTEP_Y.append(sy)
            xm1 = x - 1.
            self.XM1.append(xm1)
            self.YM1.append(xm1.T)