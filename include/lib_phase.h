#ifndef _SIMULATOR_
#define _SIMULATOR_

#include <random>
#include <complex>

#include "config.h"
#include "lib_array.h"

template <class type> 
void create_pupil_function(Array<type>& pupil, double apodize_radius){

    sizt_vector pupil_dims = pupil.get_dims();
    double xc = pupil_dims[0]/2.;
    double yc = pupil_dims[1]/2.;
    double ds = 0.0;

    for(sizt xs = 0; xs < pupil_dims[0]; xs++){
        for(sizt ys = 0; ys < pupil_dims[1]; ys++){
            ds = sqrt(pow(xs - xc, 2) + pow(ys - yc, 2));
            pupil(xs, ys) = ds < apodize_radius ? static_cast<type>(1) : static_cast<type>(0);
        }
    }
}

void create_phase_screen_fourier(Array<cmpx>& fourier, double fried, double size){

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
            amp = frq == 0.0 ? 0.0 : sqrt(0.023)*pow(size/fried, 5./6.) / pow(frq, 11./6.);
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

#endif
