import numpy as np, sys
from astropy.io import fits
from functools import partial
from matplotlib import pyplot as plt
from multiprocessing import Pool

def writefits(arr, name):
    hdu = fits.PrimaryHDU(arr)
    hdl = fits.HDUList([hdu])
    hdl.writeto(name)


def add_noise(scaling, cube):
    temp = np.zeros_like(cube)
    for i in range(cube.shape[0]):
        noise = np.random.poisson(cube[i,:,:])
        temp[i,:,:] = cube[i,:,:] + noise

    return temp

def main():

    inpname = sys.argv[1]
    outname = sys.argv[2]
    
    scaling = float(sys.argv[3])
    muram = fits.open(inpname)[0].data

    xs = muram.shape[0]
    ys = muram.shape[1]
    for i in range(xs):
        print('Iteration: ', i)
        for j in range(ys):
            muram[i,j] = np.random.poisson(muram[i,j]*scaling)
         
    writefits(muram, outname)

main()
