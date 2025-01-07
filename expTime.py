import astropy.units as u, numpy as np 
import matplotlib.pyplot as plt 
from astropy import constants as const
from astropy.table import Table 
from astropy.io import fits
import importlib


### Note: the following adapted from JT's translation of KF's simulation code
### Sent 12/14/2024

# Add some units that will make calculations with units explicit. 
# To use these later, syntax is u.Unit('shutter') or map_bin 
shutter = u.def_unit('shutter') # the unit defined for one MSA shutter 
map_bin = u.def_unit('map_bin') # the unit defined for one "bin" of the input map - so we can reserve "pixel" for the detector 
#u.add_enabled_units([shutter, map_bin]) 

instrument_file = 'UVI_v1temp_G155L_MCP_performance_071024.txt'

class MOS(object):
    def __init__(self,fov = [4.,4.] * u.arcmin,
                 shutter_size = [0.125, 0.250] * u.arcsec,
                  aperture = 6.5 * u.m ):
        
        # shutter_size: Projected MSA shutters, 125mas x 250mas, best guess

        shutter_area = shutter_size[0] * shutter_size[1]  # Projected MSA shutters, 125mas x 250mas, best guess
        print('The shutter size is: ', str(shutter_size)) 
        from hwo_cgm.data import inst
        with importlib.resources.path(inst, instrument_file) as pth:
            inst = Table.read(pth, format='csv', guess=False) 
        self.inst = inst
        self.fov = fov
        self.shutter_size = shutter_size
        self.aperture = aperture

    def get_aeff(self,obswave):
        aeff = np.interp(obswave.to(u.AA).value, self.inst['Wavelength'], self.inst['A_Eff']) * (u.cm)**2  
        aeff_aper = aeff * self.aperture**2/(6.5*u.m)**2
        print('Effective Area for this run = '+ str(aeff_aper)) 
        return aeff_aper

        
class Map(object):
    def __init__(self, dataarr, redshift, instr, units='cts', pixscale_physical = 750. * u.pc,
                cosmo=None,map_center=None,binning=1.0):
        # pixscale_physical: the physical size of an input map bin, a property of the input simulation 
        # instr: for now, just something from the MOS class
        # binning: binning factor to get acceptable S/N
        if cosmo:
            self.cosmo=cosmo
        else:
            from astropy.cosmology import Planck18
            self.cosmo = Planck18

        self.redshift = redshift * u.dimensionless_unscaled
        self.instr = instr
        self.pixscale_physical = pixscale_physical
        self.binning = binning
        self.lumdist = self.cosmo.luminosity_distance(redshift)
        self.bin_scale = pixscale_physical / self.cosmo.kpc_proper_per_arcmin(redshift) / map_bin**0.5
        self.bin_scale = self.bin_scale.to(u.arcsec/ map_bin**0.5)

        print('The bin scale on the *model map* (not the detector) is: ', self.bin_scale) 

        #TODO: add effective area bit
        #aeff_use = np.interp(obs_wave, inst['Wavelength'], inst['A_Eff']) * (u.cm)**2  
        #print('Effective Area for this run = '+ str(aeff_use)) 
        if map_center:
            self.map_center = map_center
        else:
            map_center = [int(np.shape(dataarr)[0]/2),int(np.shape(dataarr)[1]/2)]
            self.map_center = map_center

        n_el_x = int(((instr.fov[0]/2) / self.bin_scale).to(map_bin**0.5).value)   #pixels, half-width in x
        n_el_y = int(((instr.fov[1]/2) / self.bin_scale).to(map_bin**0.5).value)  #pixels, half-width in y

        trunc_map = dataarr[map_center[0]-n_el_x:map_center[0]+n_el_x,  map_center[1]-n_el_y:map_center[1]+n_el_y]
        self.trunc_map = trunc_map

        shut2bin = u.Unit('shutter') * (self.bin_scale)**2 / (instr.shutter_size[0] * instr.shutter_size[1])     # how many HUMS shutter in a map pixel?
        self.shut2bin = shut2bin

        final_res = self.bin_scale * binning # binned arcsec / resel in final map   # total resoluton in binned final map 
        self.final_res = final_res

        print('The overall field of view is: ', instr.fov) 
        print('The raw bin angular scale is: ', self.bin_scale) 
        print('The binned map resolution is: ', final_res) 
        print('The shutter to map bin conversion is: ', shut2bin) 
        print() 

    def sn(self,exptime,line='Lya',bg_shutter=0.0016*u.ph/u.s):
        #line: possible choices are 'Lya' and 'OVI'
        #bg_shutter: assumes 400um x 200um footprint of a shutter at the HUMS focal plane * an MCP background = 2 cts/cm2/s
            #per email to Haeun Chung, 05/30/24 (from K. France?)
        bg_shutter = bg_shutter/shutter #add unit 'per shutter'
        bg_bin = bg_shutter * self.shut2bin #total background in each input image bin size
        bg_binned = bg_bin * self.binning #total background in 'effective resolution size' after binning up to build S/N

        #texp = 100000. * u.s  # total exposure time [s]

        # Grab effective area at the observed wavelength of interest
        if 'Lya' in line:
            restwave = 1215.67*u.AA
        elif 'OVI' in line:
            #restwave = 1031.93*u.AA
            restwave = 1035*u.AA
        obswave = restwave * (1.+self.redshift)
        aeff = self.instr.get_aeff(obswave)

        # Compute total counts:  [ph/cm2/s/"^2]  * ["^2]/binned image * [cm2] * [s]  =  counts
        counts = self.trunc_map * (self.final_res)**2 * aeff * exptime  

        # Compute Background term (a bit of a kludge here for the units, which should work out right)
        noise_total = (counts.value + (bg_binned.value * exptime.to(u.s).value )) ** 0.5  
        

        # Compute SNR map
        snr_map = counts / noise_total 
        return snr_map

