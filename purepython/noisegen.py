# (c) Yann Thorimbert 2017
"""This module provides simple functions to generate 2D terrain or noise with
no dependency, for Python2 and Python3."""
from __future__ import print_function, division
import random, math

try:
    import pygame
    import pygame.gfxdraw as gfxdraw
except:
    print("Could not import pygame. build_surface function won't work.")

def generate_terrain(n_octaves, size, chunk=(0,0), persistance=2.):
    """
    Returns a <S> times <S> array of heigth values for <n_octaves>, using
    <chunk> as seed.
    """
    S = size
    h, min_res = _gen_hmap(n_octaves, S, chunk)
    terrain = [[0. for x in range(S)] for y in range(S)]
    res = int(S)
    step = res//min_res
    change_cell = True #indicates when polynomial coeffs have to be recomputed
    amplitude = persistance
    for i in range(n_octaves):
        delta = 1./res #size of current cell
        x_rel = 0. #x-pos in the current cell
        for x in range(S): #here x is coord of pixel
            y_rel = 0. #y-pos in the current cell
            x2 = x_rel*x_rel;
            smoothx = 3.*x2 - 2.*x_rel*x2;
            for y in range(S):
                y2 = y_rel*y_rel
                smoothy = 3.*y2 - 2.*y_rel*y2
                diag_term = x_rel*y_rel - smoothx*y_rel - smoothy*x_rel
                if change_cell:
                    idx0, idy0 = int(x/res)*step, int(y/res)*step
                    idx1, idy1 = idx0+step, idy0+step
                    h00 = h[idx0][idy0]
                    h01 = h[idx0][idy1]
                    h10 = h[idx1][idy0]
                    h11 = h[idx1][idy1]
                    #
                    dx = h10 - h00
                    dy = h01 - h00
                    A = dx - h11 + h01
                    change_cell = False
                dh = h00 + smoothx*dx + smoothy*dy + A*diag_term
                terrain[x][y] += amplitude*dh
                #
                y_rel += delta
                if y_rel >= 1.: #periodicity
                    change_cell = True
                    y_rel = 0.
            x_rel += delta
            if x_rel >= 1.: #periodicity
                change_cell = True
                x_rel = 0.
        res //= 2
        step = res//min_res
        amplitude /= persistance
    return terrain

def generate_terrain_cache(n_octaves, size, chunk=(0,0), persistance=2.):
    """
    Returns a <S> times <S> array of heigth values for <n_octaves>, using
    <mapcoord> as seed.

    Makes use of cached values. Slightly faster than not cached version.
    """
    S = size
    h, min_res = _gen_hmap(n_octaves, S, chunk)
    terrain = [[0. for x in range(S)] for y in range(S)]
    res = int(S)
    step = res//min_res
    change_cell = True #indicates when polynomial coeffs have to be recomputed
    amplitude = persistance
    for i in range(n_octaves):
        delta = 1./res #size of current cell
        x_rel = 0. #x-pos in the current cell
        for x in range(S): #here x is coord of pixel
            y_rel = 0. #y-pos in the current cell
            smoothx = SMOOTH[i][x]
            for y in range(S):
                smoothy = SMOOTH[i][y]
                diag_term = DTERM[i][x][y]
                if change_cell:
                    idx0, idy0 = int(x/res)*step, int(y/res)*step
                    idx1, idy1 = idx0+step, idy0+step
                    h00 = h[idx0][idy0]
                    h01 = h[idx0][idy1]
                    h10 = h[idx1][idy0]
                    h11 = h[idx1][idy1]
                    #
                    dx = h10 - h00
                    dy = h01 - h00
                    A = dx - h11 + h01
                    change_cell = False
                dh = h00 + smoothx*dx + smoothy*dy + A*diag_term
                terrain[x][y] += amplitude*dh
                # smoothx, diagterm
                y_rel += delta
                if y_rel >= 1.: #periodicity
                    change_cell = True
                    y_rel = 0.
            x_rel += delta
            if x_rel >= 1.: #periodicity
                change_cell = True
                x_rel = 0.
        res //= 2
        step = res//min_res
        amplitude /= persistance
    return terrain

def pix(x,y,n_octaves,h,persistance):
    """Used for local terrain generation, as in generate_terrain_local."""
    tot = 0.
    amplitude = persistance
    res = int(S)
    for i in range(n_octaves):
        smoothx = SMOOTH[i][x]
        smoothy = SMOOTH[i][y]
        diag_term = DTERM[i][x][y]
        #
        idx0,idx1 = IDX[i][x]
        idy0,idy1 = IDX[i][y]
        h00 = h[idx0][idy0]
        h01 = h[idx0][idy1]
        h10 = h[idx1][idy0]
        h11 = h[idx1][idy1]
        dx = h10 - h00
        dy = h01 - h00
        A = dx - h11 + h01
        #
        dh = h00 + smoothx*dx + smoothy*dy + A*diag_term
        tot += amplitude*dh
        res //= 2
        amplitude /= persistance
    return tot

def generate_terrain_local(n_octaves, size, chunk=(0,0), persistance=2.):
    """
    Returns a <S> times <S> array of heigth values for <n_octaves>, using
    <chunk> as seed.

    Makes use of cached values.

    This function is ~2x slower for large terrains, but faster when only a
    fraction of the terrain need to be generated.
    """
    S = size
    h, min_res = _gen_hmap(n_octaves, S, chunk)
    terrain = [[0. for x in range(S)] for y in range(S)]
    for x in range(S): #here x is coord of pixel
        for y in range(S):
            terrain[x][y] = pix(x,y,n_octaves,h,persistance)
    return terrain


class ColorScale: #tricky structure to obtain fast colormap from heightmap

    def __init__(self, colors, minval=0., default=None):
        """<colors> is on the form (c1,c2,maxval)."""
        self.colors = colors
        for i in range(len(self.colors)):
            c1,c2,maxval = self.colors[i]
            if i > 0:
                minval = self.colors[i-1][3]
            delta = maxval - minval
            self.colors[i] = [c1,c2,minval,maxval,delta]
        self.default = self.colors[0][0]

    def get(self, h):
        for c1, c2, m, M, delta in self.colors:
            if m <= h <= M:
                factor = (h-m)/delta
                kfactor = 1. - factor
                r = kfactor*c1[0] + factor*c2[0]
                g = kfactor*c1[1] + factor*c2[1]
                b = kfactor*c1[2] + factor*c2[2]
                return (r,g,b)
        return self.default



def _gen_hmap(n_octaves, S, chunk):
    """Generate random hmap used by terrain generation.
    THIS IS NOT THE ACTUAL TERRAIN GENERATION FUNCTION."""
    min_res = int(S / 2**(n_octaves-1))
    hmap_size = S//min_res + 1
    random.seed(chunk)
    h = [[random.random() for x in range(hmap_size)] for y in range(hmap_size)]
    #
    XCOORD, YCOORD = chunk
    #left
    random.seed((XCOORD,YCOORD))
    for y in range(hmap_size):
        h[0][y] = random.random()
    #right
    random.seed((XCOORD+1,YCOORD))
    for y in range(hmap_size):
        h[-1][y] = random.random()
    #top
    random.seed((XCOORD,YCOORD))
    for x in range(hmap_size):
        h[x][0] = random.random()
    #bottom
    random.seed((XCOORD,YCOORD+1))
    for x in range(hmap_size):
        h[x][-1] = random.random()
    random.seed((XCOORD,YCOORD))
    h[0][0] = random.random()
    random.seed((XCOORD+1,YCOORD+1))
    h[-1][-1] = random.random()
    random.seed((XCOORD,YCOORD+1))
    h[0][-1] = random.random()
    random.seed((XCOORD+1,YCOORD))
    h[-1][0] = random.random()
    return h, min_res



def get_cache(n_octaves, S):
    """Build cache that is used by some terrain generation functions."""
    min_res = int(S / 2**(n_octaves-1))
    res = int(S)
    step = res//min_res
    smoothx_cache, diag_term_cache, idx_cache = [], [], []
    for i in range(n_octaves):
        smoothx_cache.append([0. for x in range(S)])
        diag_term_cache.append([[0. for x in range(S)] for y in range(S)])
        idx_cache.append([0 for x in range(S)])
    for i in range(n_octaves):
        delta = 1./res #size of current cell
        x_rel = 0. #x-pos in the current cell
        for x in range(S): #here x is coord of pixel
            y_rel = 0. #y-pos in the current cell
            x2 = x_rel*x_rel
            smoothx = 3.*x2 - 2.*x_rel*x2
            smoothx_cache[i][x] = smoothx
            idx_cache[i][x] = (int(x/res)*step, int(x/res)*step + step)
            for y in range(S):
                y2 = y_rel*y_rel
                smoothy = 3.*y2 - 2.*y_rel*y2
                diag_term = x_rel*y_rel - smoothx*y_rel - smoothy*x_rel
                diag_term_cache[i][x][y] = diag_term
                y_rel += delta
                if y_rel >= 1.: #periodicity
                    y_rel = 0.
            x_rel += delta
            if x_rel >= 1.: #periodicity
                x_rel = 0.
        res //= 2
        step = res//min_res
    return smoothx_cache, diag_term_cache, idx_cache


def normalize(terrain):
    """Normalize in place the values of <terrain>."""
    M = max([max(line) for line in terrain])
    m = min([min(line) for line in terrain])
    S = len(terrain)
    for x in range(S):
        for y in range(S):
            terrain[x][y] = (terrain[x][y] - m)/(M-m)
    return terrain


def build_surface(terrain, colorscale):
    """Return a pygame Surface using <colorscale> as color scale. Suppose
    <terrain> is already normalized."""
    S = len(terrain)
    surface = pygame.Surface((S,S))
    for x in range(S):
        for y in range(S):
            color = colorscale.get(terrain[x][y])
            gfxdraw.pixel(surface,x,y,color)
    return surface