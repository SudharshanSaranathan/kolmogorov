#ifndef _SIMULATOR_
#define _SIMULATOR_

#include <random>
#include <cstring>
#include <complex>

#include "config.h"
#include "lib_array.h"

#define _USE_MATH_DEFINES

template <class type> 
void create_aperture_function(Array<type>& aperture, double aperture_radius){

/*
 * Vector declaration.
 * --------------------------------------------
 * Name             Type            Description
 * --------------------------------------------
 * aperture_dims    sizt_vector     Dimensions of the aperture
 */

    sizt_vector aperture_dims = aperture.get_dims();

/*
 * Variable declaration.
 * ------------------------------------------------
 * Name                 Type        Description
 * ------------------------------------------------
 * aperture_center_x    double      Center of aperture, abscissa.
 * aperture_center_y    double      Center of aperture, ordinate.
 * distance_center      double      Distance to the center of aperture.
 */ 

    double aperture_center_x = double(aperture_dims[0])/2.0;
    double aperture_center_y = double(aperture_dims[1])/2.0;
    double distance_center   = 0.0;
/* -----------------------
 * Make aperture function.
 * -----------------------
 */

    for(sizt xs = 0; xs < aperture_dims[0]; xs++){
        for(sizt ys = 0; ys < aperture_dims[1]; ys++){
            distance_center = sqrt(pow(xs - aperture_center_x, 2) + pow(ys - aperture_center_y, 2));
            aperture(xs, ys) = distance_center <= aperture_radius ? static_cast<type>(1) : static_cast<type>(0);
        }
    }
}

void create_phase_screen_fourier_shifted(Array<cmpx>& fourier, double fried, double sim_size){

    Array<cmpx> fourier_copy(fourier);
    sizt_vector fourier_dims = fourier.get_dims();

    double xc  = fourier_dims[0]/2;
    double yc  = fourier_dims[1]/2;
    double amp = 1.0;
    double frq = 1.0;

    std::default_random_engine generator(rand());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for(int xpix = 0; xpix < fourier_dims[0]; xpix++){
        for(int ypix = 0; ypix < fourier_dims[1]; ypix++){
            frq = sqrt(pow(xpix - xc, 2) + pow(ypix - yc, 2));
            amp = frq == 0.0 ? 0.0 : sqrt(0.023) * pow(sim_size / fried, 5. / 6.) / pow(frq, 11./6.);
            double cosphi = distribution(generator);
            double sinphi = distribution(generator);
 
            cmpx phase(amp*cosphi, amp*sinphi);
            fourier_copy(xpix, ypix) = phase;
        }
    }

    for(sizt xs = 0; xs < fourier_dims[0]; xs++){
        for(sizt ys = 0; ys < fourier_dims[1]; ys++){
            sizt xsn = (xs + (sizt)xc) % fourier_dims[0];
            sizt ysn = (ys + (sizt)yc) % fourier_dims[1];
            fourier(xs, ys) = fourier_copy(xsn, ysn);
        }
    }
}

void create_phase_screen_fourier(Array<cmpx>& fourier, double fried, double sim_size){

    sizt_vector fourier_dims = fourier.get_dims();

    double xc  = fourier_dims[0]/2;
    double yc  = fourier_dims[1]/2;
    double amp = 1.0;
    double frq = 1.0;

    std::default_random_engine generator(rand());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for(int xpix = 0; xpix < fourier_dims[0]; xpix++){
        for(int ypix = 0; ypix < fourier_dims[1]; ypix++){
            frq = sqrt(pow(xpix, 2) + pow(ypix, 2));
            amp = frq == 0.0 ? 0.0 : sqrt(0.023) * pow(sim_size / fried, 5. / 6.) / pow(frq, 11./6.);
            double cosphi = distribution(generator);
            double sinphi = distribution(generator);
 
            cmpx phase(amp*cosphi, amp*sinphi);
            fourier(xpix, ypix) = phase;
        }
    }

}

void get_residual_phase_screen(Array<double>& phase, Array<double>& basis, Array<double>& mode_cfs_weights, sizt_vector norm){

    sizt_vector dims_basis = basis.get_dims();
    sizt_vector dims_mode{dims_basis[1], dims_basis[2]};
    sizt_vector dims_mode_cfs{dims_basis[0]};

    Array<double> mode(dims_mode);
    Array<double> mode_cfs(dims_mode_cfs);

    for(sizt ind = 0; ind < dims_basis[0]; ind++){

        memcpy(mode[0], basis[ind], mode.get_size() * sizeof(double));
        mode_cfs(ind) = (phase * mode).get_total()  / norm[ind];
        phase -= mode * (mode_cfs(ind) * mode_cfs_weights(ind));

    }
}

#endif
