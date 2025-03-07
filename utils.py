from astropy import units as u
import numpy as np


def get_restwave(line):
    if ('Lya' in line)|('Lyalpha' in line):
        restwave = 1215.67 * u.AA
    elif 'OVI' in line: 
        restwave = 1031.93 * u.AA
    elif 'CIII' in line:
        restwave = 977.02 * u.AA
    elif 'CIV' in line:
        restwave = 1548.195 * u.AA

    return restwave
