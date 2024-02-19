import math
import os

# check if all bands available for a s2 uuid and grid
def have_all_bands(s2_root, uuid, s2_grid):
    s2_bands = ['b05',  'b06',  'b07',  'b08',  'b11',  'b12',  'tci']
    for band in s2_bands:
        this_path = os.path.join(s2_root, uuid, band, s2_grid+'.png')
        if not os.path.isfile(this_path):
            return False
    return True

# check if a grid is covered by NAIP's ground cover limits
def covered_by_naip(naip_grid):
    return naip_grid[0] <= 39708 and naip_grid[0] >= 20571 and naip_grid[1] <= 55878 and naip_grid[1] >= 45013

# helper functions to convert between coordinates and grids
# imported from Satlas https://github.com/allenai/satlas/blob/main/satlas/util/__init__.py
def geo_to_mercator(p, zoom=13, pixels=512):
    n = 2**zoom
    x = (p[0] + 180.0) / 360 * n
    y = (1 - math.log(math.tan(p[1] * math.pi / 180) + (1 / math.cos(p[1] * math.pi / 180))) / math.pi) / 2 * n
    return (x*pixels, y*pixels)

def mercator_to_geo(p, zoom=13, pixels=512):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)