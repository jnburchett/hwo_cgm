import astropy.units as u, numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from astropy import constants as const
from astropy.table import Table 
from astropy.io import fits
import importlib
import mpl_toolkits.axes_grid1 as axgrid



### Note: the following adapted from JT's translation of KF's simulation code
### Sent 12/14/2024

# Add some units that will make calculations with units explicit. 
# To use these later, syntax is u.Unit('shutter') or map_bin 
shutter = u.def_unit('shutter') # the unit defined for one MSA shutter 
map_bin = u.def_unit('map_bin') # the unit defined for one "bin" of the input map - so we can reserve "pixel" for the detector 
try:
    u.add_enabled_units([shutter, map_bin]) 
except:
    pass

instrument_file = 'UVI_v1temp_G155L_MCP_performance_071024.txt'

class MOS(object):
    def __init__(self,fov = [4.,4.] * u.arcmin,
                 shutter_size = [0.125, 0.250] * u.arcsec,
                  aperture = 6.5 * u.m ):
        
        # shutter_size: Projected MSA shutters, 125mas x 250mas, best guess

        shutter_area = shutter_size[0] * shutter_size[1]  # Projected MSA shutters, 125mas x 250mas, best guess
        print('Initializing MOS instrument')
        print('The shutter size is: ', str(shutter_size)) 
        print('')
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
        self.bin_scale = pixscale_physical / self.cosmo.kpc_proper_per_arcmin(redshift)/ map_bin**0.5
        self.bin_scale = self.bin_scale.to(u.arcsec/ map_bin**0.5)

        print('Initializing map with corresponding instrument.')
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
        self.map_shape_phys=np.shape(trunc_map)*self.pixscale_physical
        self.map_shape_ang=np.shape(trunc_map)*self.bin_scale*map_bin**0.5

        shut2bin = u.Unit('shutter') * (self.bin_scale)**2 / (instr.shutter_size[0] * instr.shutter_size[1])     # how many HUMS shutter in a map pixel?
        self.shut2bin = shut2bin

        final_res = self.bin_scale * binning # binned arcsec / resel in final map   # total resoluton in binned final map 
        self.final_res = final_res

        final_res_phys = self.pixscale_physical * binning # binned kpc / resel in final map
        self.final_res_phys = final_res_phys / map_bin**0.5

        self.shutter_area_phys = (self.pixscale_physical/map_bin**0.5)**2 / self.shut2bin 

        print('The overall field of view is: ', instr.fov)
        print('The overall field of view (physical) is: ', self.map_shape_phys) 
        print('The raw bin angular scale is: ', self.bin_scale) 
        print('The raw bin physical scale is: ', self.pixscale_physical/ map_bin**0.5) 
        print('The binned map resolution is: ', final_res) 
        print('The binned map physical resolution is: ', self.final_res_phys.to(u.kpc/map_bin**0.5)) 
        print('The shutter to map bin conversion is: ', shut2bin) 
        print('The shutter area in physical units is: ', self.shutter_area_phys) 
        print() 

    def sn(self,exptime,line='Lya',bg_shutter=0.0016*u.ph/u.s):
        #line: possible choices are 'Lya' and 'OVI'
        #bg_shutter: assumes 400um x 200um footprint of a shutter at the HUMS focal plane * an MCP background = 2 cts/cm2/s
            #per email to Haeun Chung, 05/30/24 (from K. France?)
        bg_shutter = bg_shutter/shutter #add unit 'per shutter'
        bg_bin = bg_shutter * self.shut2bin #total background in each input image bin size
        bg_binned = bg_bin * self.binning #total background in 'effective resolution size' after binning up to build S/N

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
    
def make_sn_map(sndata,map,line='Lya',cmap=cm.inferno,vmin=0,vmax=5,
            extent=100.*u.kpc,axislabels='distance',numticks=5,log=False):

    fig, ax1 = plt.subplots(1, 1, figsize=(11.5, 10))
    fig.subplots_adjust(bottom=0.1,top=0.95,left=0.1)
    colmap=cm.inferno

    if log:
        toplot = np.log10(sndata.value)
    else:
        toplot = sndata.value
    img = ax1.imshow(toplot,vmin=vmin,vmax=vmax,origin='lower',cmap=cmap)

    ax1.patch.set_facecolor(cmap(0.)) # sets background color to lowest color map value

    div = axgrid.make_axes_locatable(ax1)
    cax = div.append_axes("right",size="5%",pad=0.1)
    cbar = plt.colorbar(img, cax=cax,orientation='vertical')

    if 'distance' in axislabels:
        distunit = extent.unit
        extent=extent.value
        xdistarr = np.linspace(0,1,numticks)*extent - extent/2
        xtickloc = np.linspace(0,sndata.shape[0],numticks)
        xlabels = [f'{x:1.0f}' for x in xdistarr]
        ydistarr = np.linspace(0,1,numticks)*extent - extent/2
        ytickloc = np.linspace(0,sndata.shape[1],numticks)
        ylabels = [f'{y:1.0f}' for y in ydistarr]
        ax1.set_xticks(xtickloc)
        ax1.set_xticklabels(xlabels)
        ax1.set_yticks(ytickloc)
        ax1.set_yticklabels(ylabels)
        ax1.set_xlabel(distunit); ax1.set_ylabel(distunit)

    if log:
        logpart = 'log$_{10}$ '
    else:
        logpart = ''

    if 'Lya' in line:
        linepart='Ly$\\alpha$'
    else: 
        linepart=line
    clabel = r''+logpart+linepart+' S/N per bin'
    cbar.ax.set_ylabel(r'%s' % (clabel))   

    return fig

