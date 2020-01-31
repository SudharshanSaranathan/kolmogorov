#ifndef _CONFIG_
#define _CONFIG_

#include <string>
#include <limits>
#include <vector>
#include <complex>

/* ----------------
 * Define typedefs.
 * ----------------
 */

using precision = float;

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

typedef struct io_t{
public:
    
    static string read_image_from;
    static string read_fried_from;
    static string read_basis_from;
    static string read_weights_from;
    static string read_fft_psf_wisdom_from;
    static string read_fft_phase_wisdom_from;
    static string read_aperture_function_from;

    static string write_log_to;
    static string write_phase_to;
    static string write_images_to;
    static string write_residual_to;
    static string write_psf_to;
    
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
    static bool  airy_disk;

} aperture_t;

typedef struct image_t{
public:
    
    static float original_sampling;
    static float degraded_sampling;

} image_t;

int config_parse(const char*);

#endif
