import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from astropy.table import Table

import h5py

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid

from hwo_cgm import utils as hcu

def load_tng(filename,line='Lya'):
    restwave = hcu.get_restwave(line)
    with h5py.File(filename,'r') as ff:
        grid = ff['grid']
        data = grid[()]

        masked_data = np.ma.masked_invalid(data)
        data_ph = np.power(10, masked_data) * u.ph/u.s/u.cm**2/u.arcsec**2
        data_cgs = data_ph * const.h.to('erg s') * const.c.to(u.AA/u.s) / restwave/u.ph

        datadict = {'data_ph':data_ph, 'data_cgs':data_cgs, 'line': line,
                    'restwave': restwave}
    return datadict

def make_map(dataarr,line='Lya',units='cgs',cmap=cm.inferno,vmin=-21,
             vmax=-15,extent=600.*u.kpc,axislabels='distance',numticks=5):

    fig, ax1 = plt.subplots(1, 1, figsize=(11.5, 10))
    fig.subplots_adjust(bottom=0.1,top=0.95,left=0.1)
    colmap=matplotlib.cm.inferno

    logdat = np.log10(dataarr.value)
    img = ax1.imshow(logdat,vmin=vmin,vmax=vmax,origin='lower',cmap=cmap)

    ax1.patch.set_facecolor(cmap(0.)) # sets background color to lowest color map value

    div = axgrid.make_axes_locatable(ax1)
    cax = div.append_axes("right",size="5%",pad=0.1)
    cbar = plt.colorbar(img, cax=cax,orientation='vertical')

    if 'distance' in axislabels:
        distunit = extent.unit
        extent=extent.value
        xdistarr = np.linspace(0,1,numticks)*extent - extent/2
        xtickloc = np.linspace(0,dataarr.shape[0],numticks)
        xlabels = [f'{x:1.0f}' for x in xdistarr]
        ydistarr = np.linspace(0,1,numticks)*extent - extent/2
        ytickloc = np.linspace(0,dataarr.shape[1],numticks)
        ylabels = [f'{y:1.0f}' for y in ydistarr]
        ax1.set_xticks(xtickloc)
        ax1.set_xticklabels(xlabels)
        ax1.set_yticks(ytickloc)
        ax1.set_yticklabels(ylabels)
        ax1.set_xlabel(distunit); ax1.set_ylabel(distunit)
   
    if 'cgs' in units:
        unitpart='erg'
    elif 'photon' in units:
        unitpart='ph'

    if 'Lya' in line:
        linepart='Ly$\\alpha$'
    else: 
        linepart=line
    clabel = r''+linepart+' [log '+unitpart+'s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]'
    cbar.ax.set_ylabel(r'%s' % (clabel))   

    return fig

def write_fits(datadict,outfile,units='ph'):
    if 'ph' in units:
        towrite = datadict['data_ph'].value
        unitcard = 'ph/s/cm2/arcsec2'
    newhdu=fits.hdu.PrimaryHDU()
    newhdu.data=towrite
    newhdu.header['units']=unitcard
    hlist = fits.HDUList(hdus=[newhdu])
    outf = open(outfile,'wb')
    hlist.writeto(outf)
    outf.close()

def load_foggie(filename, line='Lyalpha', orientation='edge-on', redshift='0.0'):
    
    ff = h5py.File(filename, 'r')
    grid = ff['z='+redshift]
    print('Opening FOGGIE halo name:',grid.attrs['halo_name'])
    print("Groups and datasets in the HDF5 file:")
    print('FOGGIE redshift snapshots:',list(ff.keys()))
    print('')
        
    print('Metadata keys included:')
    print(grid.attrs.keys())
    print('')
    
    print('Data included:')
    print(grid.keys())
    print('')
    
    print('data grid extent:', grid.attrs['image_extent_kpc'],'kpc')

    if 'edge' in orientation:
        griddata = grid[line+'_emission_edge'][()]
    elif 'face' in orientation:
        griddata = grid[line+'_emission_face'][()]

    print('physical scale per pixel:',  grid.attrs['image_extent_kpc']/np.shape(griddata)[0],'kpc')    
    print('')

    data_ph = griddata * u.ph * 1/u.s * 1/u.cm**2 * 1/u.arcsec**2 / 4.25e10
    photon_wavelength = hcu.get_restwave(line)
    
    data_ph = griddata * u.ph * 1/u.s * 1/u.cm**2 * 1/u.arcsec**2 / 4.25e10
    data_cgs = data_ph * const.h.to('erg s') * const.c.to(u.AA/u.s) /photon_wavelength/u.ph
    
    datadict = {'data_ph':data_ph, 'data_cgs':data_cgs, 'line': line,
            'restwave': photon_wavelength}
    return datadict

