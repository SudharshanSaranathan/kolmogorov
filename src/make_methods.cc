#include "lib_array.h"
#include "fitsio.h"
#include <cstdlib>
#include <cmath>

int main(){

    sizt_vector   dims{64, 64, 64};
    Array<double> array(dims);

    for(sizt xpix = 0; xpix < dims[0]; xpix++){
        for(sizt ypix = 0; ypix < dims[1]; ypix++){
            for(sizt zpix = 0; zpix < dims[2]; zpix++){
                array(xpix, ypix, zpix) = sqrt(xpix + ypix*zpix);
            }
        }
    }

    Array<double> slice = array.get_slice(10, false);
    slice(32, 32) = 12.0;

    slice = array.get_slice(5, false);
    slice(24, 24) = 2.4323;

    fprintf(stdout, "%0.2lf %0.2lf\n", slice(24,24), array(5,24,24));
    fprintf(stdout, "%d %d\n", slice.get_owner(), array.get_owner());

    return(0);
}
