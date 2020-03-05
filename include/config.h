#ifndef _CONFIG_
#define _CONFIG_

#include <chrono>
#include <random>
#include <string>
#include <limits>
#include <vector>
#include <complex>

/* ----------------
 * Define typedefs.
 * ----------------
 */

using precision = double;

using uint   = unsigned int;
using ulng   = unsigned long;
using sizt   = std::size_t;
using cmpx   = std::complex<double>;
using string = std::string;

template <typename type>
using limits = std::numeric_limits<type>;

template <typename type>
using type_vector = std::vector<type>;
using sizt_vector = std::vector<sizt>;
using long_vector = std::vector<long>;
using uint_vector = std::vector<int>;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float>       Period;

typedef enum imgft {
    BIN  = 0,
    ANA  = 1,
    FITS = 2
} imgft;

typedef struct io_t{
public:
    
    static string rd_image_from;
    static string rd_fried_from;
    static string rd_basis_from;
    static string rd_coeff_from;
    static string rd_aperture_from;
    static string rd_psf_wisdom_from;
    static string rd_phs_wisdom_from;

    static string wr_psf_to;
    static string wr_phase_to;
    static string wr_image_to;
    static string wr_residual_to;
    
    static bool   save;
    static bool   clobber;

} io_t;

typedef struct sims_t{
public:
    
    static sizt  realizations_per_fried;
    static sizt  size_x_in_pixels;
    static sizt  size_y_in_pixels;
    static float size_in_meters;

} sims_t;

typedef struct aperture_t{
public:

    static float size;
    static float sampling_factor;
    static bool  make_airy_disk;

} aperture_t;

typedef struct image_t{
public:

    static float normalization;
    static float original_sampling;
    static float degraded_sampling;

} image_t;

int config_parse(const char*);

#endif
