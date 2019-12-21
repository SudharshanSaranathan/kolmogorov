import numpy as np
import sys

from astropy.io import fits
from scipy.interpolate import RectBivariateSpline

def writefits(arr, name):

    hdu = fits.PrimaryHDU(arr)
    hdl = fits.HDUList([hdu])
    hdl.writeto(name)

def get_aperture(size, sampling=1.0):

    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)

    xx, yy  = np.meshgrid(x, y)
    radius  = np.sqrt(xx**2. + yy**2.)
    indices = np.where(radius < size/(2*sampling))

    aperture = np.zeros((size, size))
    aperture[indices[0], indices[1]] = 1.0
    return(aperture)

def main():

    aperture_name = sys.argv[1]
    basis_name    = sys.argv[2]
    output_name   = sys.argv[3]

    aperture = fits.open(aperture_name)[0].data
    basis    = fits.open(basis_name)[0].data

    default_interpolation_factor = 3;

    x_low_res = np.linspace(-aperture.shape[0]/2,aperture.shape[0]/2, aperture.shape[0])
    y_low_res = np.linspace(-aperture.shape[0]/2,aperture.shape[0]/2, aperture.shape[0])
    
    x_high_res = np.linspace(-aperture.shape[0]/2,aperture.shape[0]/2, default_interpolation_factor*aperture.shape[0])
    y_high_res = np.linspace(-aperture.shape[0]/2,aperture.shape[0]/2, default_interpolation_factor*aperture.shape[0])

    abscissae_lr, ordinates_lr = np.meshgrid(x_low_res , y_low_res )
    abscissae_hr, ordinates_hr = np.meshgrid(x_high_res, y_high_res)

    aperture_interpolated = get_aperture(x_high_res.shape[0])
    basis_interpolated = np.zeros((basis.shape[0], x_high_res.shape[0], y_high_res.shape[0]))

    print(x_low_res.shape)
    print(y_low_res.shape)
    print(basis.shape)

    for i in range(basis.shape[0]):
        spline_basis_functions = RectBivariateSpline(x_low_res, y_low_res, basis[i])
        basis_interpolated[i]  = spline_basis_functions.ev(ordinates_hr, abscissae_hr)*aperture_interpolated

    writefits(basis_interpolated, output_name)
    writefits(aperture_interpolated, 'aperture_interpolated.fits')

main()

