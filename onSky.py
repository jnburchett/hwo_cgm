import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
from pycat import literature
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

plt.rcParams['font.family']='serif'
plt.rcParams['font.size']=21
plt.rcParams['mathtext.fontset']='stix'

def impact2physicalArea(impact):
    area = np.pi*impact**2
    return area

def impact2angularArea(impact,z):
    scale = cosmo.kpc_proper_per_arcmin(z)
    ang = impact/scale
    area = np.pi*ang**2
    return area

def area2sep(area,numsl):
    sep = 2*np.sqrt(area/np.pi/numsl)
    return sep

def rho_z_targs_to_sepPhys(impact,numsl):
    area = impact2physicalArea(impact)
    sep = area2sep(area,numsl)
    return sep

def rho_z_targs_to_sourceDensity(impact,z,numsl,units=u.deg**2):
    area = impact2angularArea(impact,z)
    sourceDens = numsl/area.to(units)
    return sourceDens


